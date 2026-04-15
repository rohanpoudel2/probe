from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from data.schema import TaskExample
from features.span_pooling import pool_named_spans


@dataclass
class TaskExtractionConfig:
    model_name: str
    layers: List[int]
    max_length: int = 1024
    pooling_mode: str = "mean"
    views: Optional[List[str]] = None
    modified_mode: str = "standard"
    prompt_prefix: Optional[str] = None
    prompt_suffix: Optional[str] = None
    output_dir: str = "task_activations"


class TaskActivationExtractor:
    """Span-aware extractor for the paper task families."""

    def __init__(self, cfg: TaskExtractionConfig):
        self.cfg = cfg
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def _tokenize_segments(self, example: TaskExample) -> Dict[str, object]:
        segments = example.build_segments()
        if self.cfg.prompt_prefix:
            segments = {"task_prompt": self.cfg.prompt_prefix, **segments}
        if self.cfg.prompt_suffix and "prompt" in segments:
            segments = dict(segments)
            segments["prompt"] = segments["prompt"] + self.cfg.prompt_suffix
        input_ids: List[int] = []
        token_spans: Dict[str, tuple[int, int]] = {}

        for name, text in segments.items():
            segment_ids = self.tokenizer.encode(text, add_special_tokens=False)
            start = len(input_ids)
            input_ids.extend(segment_ids)
            end = len(input_ids)
            token_spans[name] = (start, end)

            separator_ids = self.tokenizer.encode("\n\n", add_special_tokens=False)
            input_ids.extend(separator_ids)

        if input_ids[-len(separator_ids):] == separator_ids:
            input_ids = input_ids[:-len(separator_ids)]

        if len(input_ids) > self.cfg.max_length:
            original_length = len(input_ids)
            input_ids = input_ids[-self.cfg.max_length:]
            token_spans = self._truncate_spans(token_spans, original_length, self.cfg.max_length)

        full_span = (0, len(input_ids))
        token_spans["full_text"] = full_span

        if "answer" in token_spans:
            token_spans.setdefault("pre_answer", (0, token_spans["answer"][0]))
        if "reasoning" in token_spans:
            token_spans.update(self._reasoning_subspans(token_spans["reasoning"]))

        return {"input_ids": input_ids, "token_spans": token_spans}

    @staticmethod
    def _reasoning_subspans(span: tuple[int, int]) -> Dict[str, tuple[int, int]]:
        start, end = span
        length = end - start
        if length < 3:
            return {}
        cuts = np.linspace(start, end, num=4, dtype=int)
        names = ["reasoning_early", "reasoning_mid", "reasoning_late"]
        out: Dict[str, tuple[int, int]] = {}
        for name, left, right in zip(names, cuts[:-1], cuts[1:]):
            if left < right:
                out[name] = (int(left), int(right))
        return out

    @staticmethod
    def _truncate_spans(spans: Dict[str, tuple[int, int]], length: int, max_length: int) -> Dict[str, tuple[int, int]]:
        offset = max(0, length - max_length)
        new_spans: Dict[str, tuple[int, int]] = {}
        for name, (start, end) in spans.items():
            start -= offset
            end -= offset
            start = max(0, start)
            end = max(0, end)
            if start < end:
                new_spans[name] = (start, end)
        return new_spans

    def extract_example(self, example: TaskExample) -> Dict[int, Dict[str, np.ndarray]]:
        encoded = self._tokenize_segments(example)
        input_ids = encoded["input_ids"]
        token_spans = encoded["token_spans"]

        model_device = next(self.model.parameters()).device
        input_tensor = torch.tensor([input_ids], device=model_device)
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor, attention_mask=attention_mask, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        wanted_views = self.cfg.views or ["full_text"]
        result: Dict[int, Dict[str, np.ndarray]] = {}

        for layer in self.cfg.layers:
            hs = hidden_states[layer][0].detach().float().cpu().numpy()
            available_spans = {name: span for name, span in token_spans.items() if name in wanted_views}
            pooled = pool_named_spans(hs, available_spans, mode=self.cfg.pooling_mode)
            result[layer] = pooled
        return result

    def extract_split(self, examples: Iterable[TaskExample], split_name: str) -> None:
        outdir = Path(self.cfg.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        buffers: Dict[int, Dict[str, list]] = {layer: {} for layer in self.cfg.layers}
        labels: List[int] = []
        example_ids: List[str] = []
        question_ids: List[str] = []

        for ex in examples:
            ex_result = self.extract_example(ex)
            labels.append(ex.label)
            example_ids.append(ex.example_id)
            question_ids.append(ex.question_id or ex.example_id)
            for layer, pooled_views in ex_result.items():
                for view_name, vec in pooled_views.items():
                    buffers[layer].setdefault(view_name, []).append(vec)

        for layer, view_map in buffers.items():
            arrays = {view: np.stack(vs) for view, vs in view_map.items()}
            arrays["labels"] = np.asarray(labels, dtype=np.int64)
            arrays["example_ids"] = np.asarray(example_ids)
            arrays["question_ids"] = np.asarray(question_ids)
            suffix = ""
            if self.cfg.modified_mode != "standard":
                suffix = f"_{self.cfg.modified_mode}"
            np.savez_compressed(outdir / f"{split_name}_layer{layer}{suffix}.npz", **arrays)
