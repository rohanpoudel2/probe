from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
    model_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    max_length: int = 1024
    pooling_mode: str = "mean"
    views: Optional[List[str]] = None
    modified_mode: str = "standard"
    prompt_prefix: Optional[str] = None
    prompt_suffix: Optional[str] = None
    output_dir: str = "task_activations"
    use_chat_template: bool = True
    missing_view_policy: str = "error"
    require_model_generated: bool = True
    device: str = "auto"
    dataset_sha256: Optional[str] = None
    code_revision: Optional[str] = None
    code_dirty: bool = False
    split_seed: int = 42


class TaskActivationExtractor:
    """Chat-faithful span extractor for labeled, model-generated rollouts.

    Configured layer numbers are zero-based transformer block indices. Hugging
    Face returns the embedding state at ``hidden_states[0]``, so block ``L`` is
    read from ``hidden_states[L + 1]``.
    """

    _ASSISTANT_SEGMENTS = {"reasoning", "pre_answer", "answer"}

    def __init__(self, cfg: TaskExtractionConfig):
        self.cfg = cfg
        if cfg.missing_view_policy not in {"error", "drop"}:
            raise ValueError("missing_view_policy must be 'error' or 'drop'")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cli.common import resolve_torch_device

        tokenizer_revision = cfg.tokenizer_revision or cfg.model_revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            revision=tokenizer_revision,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        resolved_device = resolve_torch_device(cfg.device)
        model_kwargs = {
            "revision": cfg.model_revision,
            "torch_dtype": "auto",
        }
        if resolved_device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            **model_kwargs,
        )
        if resolved_device != "auto":
            self.model = self.model.to(resolved_device)
        self.model.eval()
        self.chat_template_sha256 = (
            hashlib.sha256(str(self.tokenizer.chat_template).encode("utf-8")).hexdigest()
            if getattr(self.tokenizer, "chat_template", None)
            else None
        )
        self._validate_layers()

    def _validate_layers(self) -> None:
        n_layers = int(getattr(self.model.config, "num_hidden_layers", -1))
        invalid = [layer for layer in self.cfg.layers if layer < 0 or (n_layers >= 0 and layer >= n_layers)]
        if invalid:
            raise ValueError(
                f"Configured transformer-block layers {invalid} are invalid for a model with {n_layers} blocks"
            )

    def _prepared_segments(self, example: TaskExample) -> Dict[str, str]:
        if self.cfg.require_model_generated and example.metadata.get("data_origin") != "on_policy_generation":
            raise ValueError(
                f"Example {example.example_id} is not marked as on-policy model generation; "
                "authored completions are prohibited in the main extraction path"
            )
        if (
            self.cfg.require_model_generated
            and example.metadata.get("eligible_for_main_study") is False
        ):
            raise ValueError(
                f"Example {example.example_id} is explicitly ineligible for the main study"
            )
        segments = dict(example.build_segments())
        if self.cfg.prompt_prefix:
            segments = {"task_prompt": self.cfg.prompt_prefix, **segments}
        if self.cfg.prompt_suffix and "prompt" in segments:
            segments["prompt"] = segments["prompt"] + self.cfg.prompt_suffix
        return {name: text for name, text in segments.items() if isinstance(text, str) and text}

    @staticmethod
    def _char_span_to_token_span(
        offsets: List[tuple[int, int]], char_start: int, char_end: int
    ) -> tuple[int, int]:
        indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end and end > start
        ]
        if not indices:
            raise ValueError(f"No tokens overlap character span {(char_start, char_end)}")
        return indices[0], indices[-1] + 1

    def _chat_messages_and_parts(
        self, example: TaskExample, segments: Dict[str, str]
    ) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
        if example.messages:
            messages = [dict(message) for message in example.messages]
            if not all(
                message.get("role") in {"system", "user", "assistant"}
                and isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError(f"Example {example.example_id} has invalid chat messages")
            # Exact named spans still come from the structured segments. They
            # must occur literally in the stored, model-generated transcript.
            return messages, list(segments.items())

        system_parts = [(name, text) for name, text in segments.items() if name == "task_prompt"]
        user_parts = [
            (name, text)
            for name, text in segments.items()
            if name not in self._ASSISTANT_SEGMENTS and name != "task_prompt"
        ]
        assistant_parts = [
            (name, text) for name, text in segments.items() if name in self._ASSISTANT_SEGMENTS
        ]
        messages: list[dict[str, str]] = []
        ordered_parts: list[tuple[str, str]] = []
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(text for _, text in system_parts)})
            ordered_parts.extend(system_parts)
        if user_parts:
            messages.append({"role": "user", "content": "\n\n".join(text for _, text in user_parts)})
            ordered_parts.extend(user_parts)
        if assistant_parts:
            messages.append(
                {"role": "assistant", "content": "\n\n".join(text for _, text in assistant_parts)}
            )
            ordered_parts.extend(assistant_parts)
        return messages, ordered_parts

    def _tokenize_chat(self, example: TaskExample, segments: Dict[str, str]) -> Dict[str, object]:
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                f"Tokenizer for {self.cfg.model_name} has no chat template; use --no_chat_template only for a pre-registered base-model study"
            )
        messages, ordered_parts = self._chat_messages_and_parts(example, segments)
        if not messages:
            raise ValueError(f"Example {example.example_id} produced no chat messages")
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=messages[-1]["role"] != "assistant",
        )
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        if "offset_mapping" not in encoded:
            raise ValueError("A fast tokenizer with offset mappings is required for exact chat spans")
        input_ids = list(encoded["input_ids"])
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]

        token_spans: Dict[str, tuple[int, int]] = {}
        cursor = 0
        for name, text in ordered_parts:
            char_start = rendered.find(text, cursor)
            if char_start < 0:
                raise ValueError(
                    f"Could not locate segment {name!r} in chat-formatted example {example.example_id}"
                )
            char_end = char_start + len(text)
            token_spans[name] = self._char_span_to_token_span(offsets, char_start, char_end)
            cursor = char_end
        return {"input_ids": input_ids, "token_spans": token_spans}

    def _tokenize_raw(self, segments: Dict[str, str]) -> Dict[str, object]:
        input_ids: List[int] = []
        token_spans: Dict[str, tuple[int, int]] = {}
        separator_ids = self.tokenizer.encode("\n\n", add_special_tokens=False)
        for name, text in segments.items():
            segment_ids = self.tokenizer.encode(text, add_special_tokens=False)
            start = len(input_ids)
            input_ids.extend(segment_ids)
            token_spans[name] = (start, len(input_ids))
            input_ids.extend(separator_ids)
        if separator_ids and input_ids[-len(separator_ids) :] == separator_ids:
            input_ids = input_ids[: -len(separator_ids)]
        return {"input_ids": input_ids, "token_spans": token_spans}

    def _tokenize_segments(self, example: TaskExample) -> Dict[str, object]:
        segments = self._prepared_segments(example)
        encoded = (
            self._tokenize_chat(example, segments)
            if self.cfg.use_chat_template
            else self._tokenize_raw(segments)
        )
        input_ids = encoded["input_ids"]
        token_spans = encoded["token_spans"]
        if not input_ids:
            raise ValueError(f"Example {example.example_id} tokenized to an empty sequence")

        if len(input_ids) > self.cfg.max_length:
            original_length = len(input_ids)
            input_ids = input_ids[-self.cfg.max_length :]
            token_spans = self._truncate_spans(token_spans, original_length, self.cfg.max_length)

        token_spans["full_text"] = (0, len(input_ids))
        if "answer" in token_spans:
            token_spans.setdefault("pre_answer", (0, token_spans["answer"][0]))
        if "reasoning" in token_spans:
            token_spans.update(self._reasoning_subspans(token_spans["reasoning"]))

        wanted = self.cfg.views or ["full_text"]
        missing = sorted(set(wanted).difference(token_spans))
        if missing:
            raise ValueError(
                f"Example {example.example_id} is missing requested views after tokenization/truncation: {missing}"
            )
        return {"input_ids": input_ids, "token_spans": token_spans}

    @staticmethod
    def _reasoning_subspans(span: tuple[int, int]) -> Dict[str, tuple[int, int]]:
        start, end = span
        if end - start < 3:
            return {}
        cuts = np.linspace(start, end, num=4, dtype=int)
        names = ["reasoning_early", "reasoning_mid", "reasoning_late"]
        return {
            name: (int(left), int(right))
            for name, left, right in zip(names, cuts[:-1], cuts[1:])
            if left < right
        }

    @staticmethod
    def _truncate_spans(
        spans: Dict[str, tuple[int, int]], length: int, max_length: int
    ) -> Dict[str, tuple[int, int]]:
        offset = max(0, length - max_length)
        new_spans: Dict[str, tuple[int, int]] = {}
        for name, (start, end) in spans.items():
            shifted_start = max(0, start - offset)
            shifted_end = max(0, end - offset)
            if shifted_start < shifted_end:
                new_spans[name] = (shifted_start, shifted_end)
        return new_spans

    def extract_example(self, example: TaskExample) -> Dict[int, Dict[str, np.ndarray]]:
        encoded = self._tokenize_segments(example)
        input_ids = encoded["input_ids"]
        token_spans = encoded["token_spans"]
        model_device = next(self.model.parameters()).device
        input_tensor = torch.tensor([input_ids], device=model_device)
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states
        wanted_views = self.cfg.views or ["full_text"]
        result: Dict[int, Dict[str, np.ndarray]] = {}
        for block_index in self.cfg.layers:
            hidden_state_index = block_index + 1
            if hidden_state_index >= len(hidden_states):
                raise ValueError(
                    f"Block {block_index} maps to hidden state {hidden_state_index}, but model returned {len(hidden_states)} states"
                )
            activations = hidden_states[hidden_state_index][0].detach().float().cpu().numpy()
            spans = {name: token_spans[name] for name in wanted_views}
            result[block_index] = pool_named_spans(
                activations, spans, mode=self.cfg.pooling_mode
            )
        return result

    def extract_split(self, examples: Iterable[TaskExample], split_name: str) -> None:
        outdir = Path(self.cfg.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        buffers: Dict[int, Dict[str, list[np.ndarray]]] = {
            layer: {} for layer in self.cfg.layers
        }
        labels: List[int] = []
        example_ids: List[str] = []
        question_ids: List[str] = []
        dropped_ids: List[str] = []

        for example in examples:
            try:
                result = self.extract_example(example)
            except ValueError:
                if self.cfg.missing_view_policy == "drop":
                    dropped_ids.append(example.example_id)
                    continue
                raise
            labels.append(example.label)
            example_ids.append(example.example_id)
            question_ids.append(example.question_id or example.example_id)
            for layer, pooled_views in result.items():
                for view_name, vector in pooled_views.items():
                    buffers[layer].setdefault(view_name, []).append(vector)

        if not labels:
            raise ValueError(f"No extractable examples remained for split {split_name}")
        wanted_views = self.cfg.views or ["full_text"]
        for layer, view_map in buffers.items():
            for view in wanted_views:
                if len(view_map.get(view, [])) != len(labels):
                    raise RuntimeError(
                        f"Layer {layer} view {view} has {len(view_map.get(view, []))} vectors for {len(labels)} labels"
                    )
            arrays = {view: np.stack(view_map[view]) for view in wanted_views}
            arrays["labels"] = np.asarray(labels, dtype=np.int64)
            arrays["example_ids"] = np.asarray(example_ids)
            arrays["question_ids"] = np.asarray(question_ids)
            arrays["feature_schema_version"] = np.asarray("2")
            arrays["layer_index_semantics"] = np.asarray("zero_based_transformer_block_output")
            arrays["model_name"] = np.asarray(self.cfg.model_name)
            arrays["model_revision"] = np.asarray(self.cfg.model_revision or "unpinned")
            arrays["tokenizer_revision"] = np.asarray(
                self.cfg.tokenizer_revision or self.cfg.model_revision or "unpinned"
            )
            arrays["chat_template_used"] = np.asarray(self.cfg.use_chat_template)
            arrays["chat_template_sha256"] = np.asarray(self.chat_template_sha256 or "none")
            arrays["dataset_sha256"] = np.asarray(self.cfg.dataset_sha256 or "unknown")
            arrays["code_revision"] = np.asarray(self.cfg.code_revision or "unknown")
            arrays["code_dirty"] = np.asarray(self.cfg.code_dirty)
            arrays["split_seed"] = np.asarray(self.cfg.split_seed)
            arrays["extraction_config_sha256"] = np.asarray(
                hashlib.sha256(
                    json.dumps(
                        {
                            "model_name": self.cfg.model_name,
                            "model_revision": self.cfg.model_revision,
                            "tokenizer_revision": self.cfg.tokenizer_revision,
                            "layers": self.cfg.layers,
                            "max_length": self.cfg.max_length,
                            "pooling_mode": self.cfg.pooling_mode,
                            "views": wanted_views,
                            "modified_mode": self.cfg.modified_mode,
                            "use_chat_template": self.cfg.use_chat_template,
                            "split_seed": self.cfg.split_seed,
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
            )
            arrays["dropped_example_ids"] = np.asarray("\n".join(dropped_ids))
            suffix = "" if self.cfg.modified_mode == "standard" else f"_{self.cfg.modified_mode}"
            np.savez_compressed(outdir / f"{split_name}_layer{layer}{suffix}.npz", **arrays)
