from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from data.outcomes import is_valid_model_outcome
from data.trajectory_schema import (
    TRAJECTORY_BASIS,
    TRAJECTORY_PROMPT_END_VIEW,
    build_trajectory_prefix_views,
    parse_trajectory_prefix_view,
    trajectory_prefix_stack_view,
)
from data.schema import TaskExample
from features.span_pooling import pool_named_spans


@dataclass
class TaskExtractionConfig:
    model_name: str
    layers: List[int]
    model_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    max_length: int = 1024
    allow_truncation: bool = False
    pooling_mode: str = "mean"
    views: Optional[List[str]] = None
    trajectory_prefix_stack_view: bool = False
    output_dir: str = "task_activations"
    use_chat_template: bool = True
    missing_view_policy: str = "error"
    require_model_generated: bool = True
    device: str = "auto"
    dataset_sha256: Optional[str] = None
    code_revision: Optional[str] = None
    code_dirty: bool = False
    split_seed: int = 42
    trajectory_prefix_percentiles: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("layers must contain at least one transformer block index")
        if self.max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if self.trajectory_prefix_stack_view and not self.trajectory_prefix_percentiles:
            raise ValueError(
                "trajectory_prefix_stack_view requires trajectory_prefix_percentiles"
            )
        if self.trajectory_prefix_percentiles is not None:
            if not self.trajectory_prefix_percentiles:
                raise ValueError("trajectory_prefix_percentiles cannot be empty when provided")
            for percentile in self.trajectory_prefix_percentiles:
                if not isinstance(percentile, int) or not (1 <= percentile <= 100):
                    raise ValueError(
                        "trajectory_prefix_percentiles must contain integers in [1, 100]"
                    )


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
        from transformers import AutoModel, AutoTokenizer
        from cli.common import inference_dtype_for_device, resolve_torch_device

        tokenizer_revision = cfg.tokenizer_revision or cfg.model_revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            revision=tokenizer_revision,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        resolved_device = resolve_torch_device(cfg.device)
        self.resolved_device = resolved_device
        model_kwargs = {
            "revision": cfg.model_revision,
            "torch_dtype": inference_dtype_for_device(resolved_device),
            "low_cpu_mem_usage": True,
        }
        # Only the transformer-block hidden states are needed; loading the base
        # model (no language-modeling head) yields identical ``hidden_states``
        # while skipping the full-vocabulary output projection on every forward.
        self.model = AutoModel.from_pretrained(
            cfg.model_name,
            **model_kwargs,
        )
        self.model = self.model.to(resolved_device)
        self.model.eval()
        self.model.requires_grad_(False)
        first_parameter = next(self.model.parameters())
        self.model_parameter_device = str(first_parameter.device)
        self.model_parameter_dtype = str(first_parameter.dtype)
        self.chat_template_sha256 = (
            hashlib.sha256(str(self.tokenizer.chat_template).encode("utf-8")).hexdigest()
            if getattr(self.tokenizer, "chat_template", None)
            else None
        )
        self._validate_layers()
        self._truncate_unused_blocks()

    def _validate_layers(self) -> None:
        n_layers = int(getattr(self.model.config, "num_hidden_layers", -1))
        invalid = [
            layer
            for layer in self.cfg.layers
            if layer < 0 or (n_layers >= 0 and layer >= n_layers)
        ]
        if invalid:
            raise ValueError(
                f"Configured transformer-block layers {invalid} are invalid for a model with {n_layers} blocks"
            )

    def _truncate_unused_blocks(self) -> None:
        """Drop transformer blocks above the highest requested layer.

        Extraction only reads ``hidden_states`` up to ``max(layers)``; any blocks
        above it are computed and thrown away. For a standard decoder stack whose
        single final norm produces ``last_hidden_state`` (e.g. ``Qwen3Model``),
        removing the unused tail and neutralising that final norm leaves every
        retained block's captured hidden state bit-identical — the top captured
        entry is post-norm, so with an identity norm it equals the raw block
        output, matching the untruncated model's intermediate state at that
        index — while skipping the upper-network forward entirely. Models that do
        not expose the expected ``.layers``/``.norm`` layout are left untouched.
        """
        base = getattr(self.model, "model", self.model)
        if not (hasattr(base, "layers") and hasattr(base, "norm")):
            return
        n_layers = int(getattr(base.config, "num_hidden_layers", len(base.layers)))
        n_needed = max(self.cfg.layers) + 1
        if n_needed >= n_layers:
            return
        base.layers = base.layers[:n_needed]
        base.norm = torch.nn.Identity()
        base.config.num_hidden_layers = n_needed

    @staticmethod
    def _normalize_views(raw_views: Optional[List[str]]) -> list[str]:
        views = raw_views or ["full_text"]
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_view in views:
            view = str(raw_view).strip()
            if not view or view in seen:
                continue
            normalized.append(view)
            seen.add(view)
        if not normalized:
            raise ValueError("at least one activation view is required")
        return normalized

    def _requested_output_views(self, token_count: int) -> list[str]:
        views = self._normalize_views(self.cfg.views)
        if (
            self.cfg.trajectory_prefix_percentiles
            and TRAJECTORY_PROMPT_END_VIEW not in views
        ):
            views.append(TRAJECTORY_PROMPT_END_VIEW)
        for view_name, _ in build_trajectory_prefix_views(
            self.cfg.trajectory_prefix_percentiles, token_count
        ):
            if view_name not in views:
                views.append(view_name)
        if self.cfg.trajectory_prefix_stack_view:
            for trajectory_view, _ in build_trajectory_prefix_views(
                self.cfg.trajectory_prefix_percentiles, token_count
            ):
                percentile_value = parse_trajectory_prefix_view(trajectory_view)
                if percentile_value is None:
                    raise RuntimeError(
                        f"Malformed trajectory view: {trajectory_view}"
                    )
                stack_view = trajectory_prefix_stack_view(percentile_value)
                if stack_view not in views:
                    views.append(stack_view)
        return views

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
            (name, text)
            for name, text in segments.items()
            if name in self._ASSISTANT_SEGMENTS
        ]
        messages: list[dict[str, str]] = []
        ordered_parts: list[tuple[str, str]] = []
        if system_parts:
            messages.append(
                {"role": "system", "content": "\n\n".join(text for _, text in system_parts)}
            )
            ordered_parts.extend(system_parts)
        if user_parts:
            messages.append(
                {"role": "user", "content": "\n\n".join(text for _, text in user_parts)}
            )
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
            raise ValueError(
                "A fast tokenizer with offset mappings is required for exact chat spans"
            )
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
        if messages[-1]["role"] == "assistant":
            assistant_text = messages[-1]["content"]
            assistant_start = rendered.rfind(assistant_text)
            if assistant_start < 0:
                raise ValueError(
                    f"Could not locate assistant response in example {example.example_id}"
                )
            token_spans["assistant_response"] = self._char_span_to_token_span(
                offsets,
                assistant_start,
                assistant_start + len(assistant_text),
            )
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
        assistant_spans = [
            token_spans[name]
            for name in self._ASSISTANT_SEGMENTS
            if name in token_spans
        ]
        if assistant_spans:
            token_spans["assistant_response"] = (
                min(start for start, _ in assistant_spans),
                max(end for _, end in assistant_spans),
            )
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
            raise ValueError(f"Example {example.example_id} is tokenized to an empty sequence")

        original_length = len(input_ids)
        truncated = original_length > self.cfg.max_length
        if truncated:
            if not self.cfg.allow_truncation:
                raise ValueError(
                    f"Example {example.example_id} has {original_length} tokens, "
                    f"exceeding max_length={self.cfg.max_length}; truncation is "
                    "prohibited unless explicitly enabled for a non-confirmatory run"
                )
            input_ids = input_ids[-self.cfg.max_length :]
            token_spans = self._truncate_spans(token_spans, original_length, self.cfg.max_length)

        token_spans["full_text"] = (0, len(input_ids))
        if "answer" in token_spans:
            token_spans.setdefault("pre_answer", (0, token_spans["answer"][0]))
        if "reasoning" in token_spans:
            token_spans.update(self._reasoning_subspans(token_spans["reasoning"]))
        assistant_span = token_spans.get("assistant_response")
        if self.cfg.trajectory_prefix_percentiles and assistant_span is None:
            raise ValueError(
                f"Example {example.example_id} lacks an assistant-response span "
                "required for trajectory extraction"
            )
        if assistant_span is not None and self.cfg.trajectory_prefix_percentiles:
            response_start, response_end = assistant_span
            if response_start <= 0:
                raise ValueError(
                    f"Example {example.example_id} has no prompt token before its response"
                )
            token_spans[TRAJECTORY_PROMPT_END_VIEW] = (
                response_start - 1,
                response_start,
            )
            response_tokens = response_end - response_start
            for view_name, window in build_trajectory_prefix_views(
                self.cfg.trajectory_prefix_percentiles, response_tokens
            ):
                token_spans[view_name] = (
                    response_start,
                    response_start + window,
                )
        if self.cfg.trajectory_prefix_stack_view:
            for trajectory_view, _ in build_trajectory_prefix_views(
                self.cfg.trajectory_prefix_percentiles,
                0 if assistant_span is None else assistant_span[1] - assistant_span[0],
            ):
                stack_view = trajectory_prefix_stack_view(
                    parse_trajectory_prefix_view(trajectory_view)
                )
                token_spans[stack_view] = token_spans[trajectory_view]

        wanted = self._requested_output_views(len(input_ids))
        missing = sorted(set(wanted).difference(token_spans))
        if missing:
            raise ValueError(
                f"Example {example.example_id} is missing requested views after tokenization/truncation: {missing}"
            )

        return {
            "input_ids": input_ids,
            "token_spans": token_spans,
            "original_token_count": original_length,
            "token_count": len(input_ids),
            "truncated": truncated,
        }

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

    def _ordered_trajectory_prefix_views(self, token_count: int) -> list[str]:
        if self.cfg.trajectory_prefix_percentiles is None:
            return []
        return [
            view_name
            for view_name, _ in build_trajectory_prefix_views(
                self.cfg.trajectory_prefix_percentiles, token_count
            )
        ]

    def _add_stack_features(
        self, pooled_views: Dict[str, np.ndarray], token_count: int
    ) -> None:
        if not self.cfg.trajectory_prefix_stack_view:
            return
        prefix_views = self._ordered_trajectory_prefix_views(token_count)
        if not prefix_views:
            return
        cumulative = []
        for view_name in prefix_views:
            if view_name not in pooled_views:
                raise ValueError(
                    f"Missing {view_name} trajectory prefix vector for stacked trajectory features"
                )
            cumulative.append(pooled_views[view_name])
            percentile = parse_trajectory_prefix_view(view_name)
            if percentile is None:
                raise ValueError(f"Malformed trajectory prefix view {view_name}")
            stacked_name = trajectory_prefix_stack_view(percentile)
            pooled_views[stacked_name] = np.concatenate(cumulative, axis=0)

    def extract_example_with_metadata(
        self, example: TaskExample
    ) -> tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, object]]:
        encoded = self._tokenize_segments(example)
        input_ids = encoded["input_ids"]
        token_spans = encoded["token_spans"]
        token_count = len(input_ids)
        wanted_views = self._requested_output_views(token_count)
        model_device = next(self.model.parameters()).device
        input_tensor = torch.tensor([input_ids], device=model_device)
        attention_mask = torch.ones_like(input_tensor)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            for block_index in self.cfg.layers:
                hidden_state_index = block_index + 1
                if hidden_state_index >= len(hidden_states):
                    raise ValueError(
                        f"Block {block_index} maps to hidden state {hidden_state_index}, but model returned {len(hidden_states)} states"
                    )
            # Gather every requested block and copy to host in one device->host
            # transfer instead of one per layer, minimising MPS synchronisations.
            block_activations = (
                torch.stack(
                    [hidden_states[block_index + 1][0] for block_index in self.cfg.layers]
                )
                .float()
                .cpu()
                .numpy()
            )
        result: Dict[int, Dict[str, np.ndarray]] = {}
        for offset, block_index in enumerate(self.cfg.layers):
            spans = {name: token_spans[name] for name in wanted_views}
            result[block_index] = pool_named_spans(
                block_activations[offset], spans, mode=self.cfg.pooling_mode
            )
            self._add_stack_features(result[block_index], token_count)
        metadata = {
            "original_token_count": int(encoded["original_token_count"]),
            "token_count": int(encoded["token_count"]),
            "truncated": bool(encoded["truncated"]),
            "token_spans": {
                name: [int(start), int(end)]
                for name, (start, end) in token_spans.items()
                if name in wanted_views
            },
            "trajectory_prefix_percentiles": (
                list(self.cfg.trajectory_prefix_percentiles)
                if self.cfg.trajectory_prefix_percentiles is not None
                else None
            ),
            "trajectory_prefix_stack_view": self.cfg.trajectory_prefix_stack_view,
            "trajectory_basis": (
                TRAJECTORY_BASIS
                if self.cfg.trajectory_prefix_percentiles
                else None
            ),
        }
        return result, metadata

    def extract_split(self, examples: Iterable[TaskExample], split_name: str) -> None:
        outdir = Path(self.cfg.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        buffers: Dict[int, Dict[str, list[np.ndarray]]] = {
            layer: {} for layer in self.cfg.layers
        }
        labels: List[int] = []
        example_ids: List[str] = []
        question_ids: List[str] = []
        annotation_outcome_classes: List[str] = []
        dropped_ids: List[str] = []
        original_token_counts: List[int] = []
        token_counts: List[int] = []
        truncation_flags: List[bool] = []
        token_spans_json: List[str] = []

        for example in examples:
            try:
                result, extraction_metadata = self.extract_example_with_metadata(example)
            except ValueError:
                if self.cfg.missing_view_policy == "drop":
                    dropped_ids.append(example.example_id)
                    continue
                raise
            labels.append(example.label)
            example_ids.append(example.example_id)
            question_ids.append(example.question_id or example.example_id)
            outcome_class = str(
                example.metadata.get("annotation_outcome_class") or ""
            ).strip()
            if (
                example.task_family != "reference_traffic"
                and not is_valid_model_outcome(outcome_class)
            ):
                raise ValueError(
                    f"Example {example.example_id} lacks a valid annotation_outcome_class; "
                    "relabel and merge this dataset with the current protocol"
                )
            annotation_outcome_classes.append(outcome_class)
            original_token_counts.append(
                int(extraction_metadata["original_token_count"])
            )
            token_counts.append(int(extraction_metadata["token_count"]))
            truncation_flags.append(bool(extraction_metadata["truncated"]))
            token_spans_json.append(
                json.dumps(extraction_metadata["token_spans"], sort_keys=True)
            )
            for layer, pooled_views in result.items():
                for view_name, vector in pooled_views.items():
                    buffers[layer].setdefault(view_name, []).append(vector)

        if not labels:
            raise ValueError(f"No extractable examples remained for split {split_name}")
        # Recompute to avoid assuming all examples share identical trajectory views.
        wanted_views = self._requested_output_views(int(np.max(token_counts)) if token_counts else 1)
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
            arrays["annotation_outcome_class"] = np.asarray(
                annotation_outcome_classes
            )
            arrays["original_token_counts"] = np.asarray(
                original_token_counts, dtype=np.int64
            )
            arrays["token_counts"] = np.asarray(token_counts, dtype=np.int64)
            arrays["truncated"] = np.asarray(truncation_flags, dtype=bool)
            arrays["token_spans_json"] = np.asarray(token_spans_json)
            arrays["feature_schema_version"] = np.asarray("3")
            arrays["layer_index_semantics"] = np.asarray("zero_based_transformer_block_output")
            arrays["model_name"] = np.asarray(self.cfg.model_name)
            arrays["model_revision"] = np.asarray(self.cfg.model_revision or "unpinned")
            arrays["tokenizer_revision"] = np.asarray(
                self.cfg.tokenizer_revision or self.cfg.model_revision or "unpinned"
            )
            arrays["chat_template_used"] = np.asarray(self.cfg.use_chat_template)
            arrays["chat_template_sha256"] = np.asarray(self.chat_template_sha256 or "none")
            arrays["pooling_mode"] = np.asarray(self.cfg.pooling_mode)
            arrays["requested_views_json"] = np.asarray(json.dumps(wanted_views, sort_keys=True))
            arrays["trajectory_prefix_percentiles"] = np.asarray(
                json.dumps(self.cfg.trajectory_prefix_percentiles or [])
            )
            arrays["trajectory_basis"] = np.asarray(
                TRAJECTORY_BASIS
                if self.cfg.trajectory_prefix_percentiles
                else "none"
            )
            arrays["max_length"] = np.asarray(self.cfg.max_length)
            arrays["allow_truncation"] = np.asarray(self.cfg.allow_truncation)
            arrays["missing_view_policy"] = np.asarray(self.cfg.missing_view_policy)
            arrays["require_model_generated"] = np.asarray(
                self.cfg.require_model_generated
            )
            arrays["resolved_device"] = np.asarray(self.resolved_device)
            arrays["model_parameter_device"] = np.asarray(
                self.model_parameter_device
            )
            arrays["model_parameter_dtype"] = np.asarray(
                self.model_parameter_dtype
            )
            arrays["feature_dtype"] = np.asarray(str(arrays[wanted_views[0]].dtype))
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
                            "allow_truncation": self.cfg.allow_truncation,
                            "pooling_mode": self.cfg.pooling_mode,
                            "views": wanted_views,
                            "trajectory_prefix_percentiles": self.cfg.trajectory_prefix_percentiles,
                            "trajectory_basis": (
                                TRAJECTORY_BASIS
                                if self.cfg.trajectory_prefix_percentiles
                                else None
                            ),
                            "use_chat_template": self.cfg.use_chat_template,
                            "missing_view_policy": self.cfg.missing_view_policy,
                            "require_model_generated": self.cfg.require_model_generated,
                            "split_seed": self.cfg.split_seed,
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
            )
            arrays["dropped_example_ids"] = np.asarray("\n".join(dropped_ids))
            np.savez_compressed(outdir / f"{split_name}_layer{layer}.npz", **arrays)
