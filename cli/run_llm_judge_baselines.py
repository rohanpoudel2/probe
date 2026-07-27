from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cli.run_task_sweep import (
    _atomic_write_jsonl,
    _load_existing_run_ids,
    _metric_payload,
    _prediction_path,
    _prediction_records,
)
from data.group_splitting import declared_protocol_split
from data.llm_judge import (
    LLM_JUDGE_CACHE_SCHEMA_VERSION,
    build_judge_messages,
    judge_context_hash,
)
from data.llm_judge_cache import atomic_save_judge_cache, load_judge_cache
from data.rollout_schema import canonical_json
from data.text_views import (
    ALLOWED_TEXT_VIEWS,
    examples_to_text_arrays,
    monitored_model_identity,
)
from evaluation.metrics import (
    alert_rate_summary,
    compute_alert_rate,
    require_disjoint_reference_groups,
    require_independent_reference_groups,
    select_threshold_at_alert_rate,
)
from task_benchmark.sampling import FewShotSelection, sample_few_shot_train
from tasks import TASK_REGISTRY
from cli.common import (
    bounded_batch_size_for_device,
    inference_dtype_for_device,
    resolve_torch_device,
)


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
JUDGE_IMPLEMENTATION_FILES = (
    Path(__file__),
    Path(__file__).parents[1] / "data" / "llm_judge.py",
    Path(__file__).parents[1] / "data" / "llm_judge_cache.py",
)


def _git_state() -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError) as err:
        raise RuntimeError("LLM-judge scoring requires Git provenance") from err
    return revision, dirty


def _implementation_sha256() -> str:
    payload = {
        path.relative_to(Path(__file__).parents[1]).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in JUDGE_IMPLEMENTATION_FILES
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _load_judge_spec(path: Path, key: str) -> tuple[dict[str, Any], str, str]:
    config_bytes = path.read_bytes()
    config = yaml.safe_load(config_bytes) or {}
    if config.get("schema_version") != "llm-judge-model-lock-v1":
        raise ValueError("Unsupported LLM-judge model-lock schema")
    models = config.get("models") or {}
    if key not in models:
        raise ValueError(f"Unknown LLM-judge key {key!r}")
    spec = dict(models[key])
    required = {
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "family",
        "max_length",
        "padding_side",
        "trust_remote_code",
        "negative_label",
        "positive_label",
        "chat_template_kwargs",
        "system_prompt",
        "protocol_role",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"LLM-judge model lock {key!r} is missing {missing}")
    for field in ("model_revision", "tokenizer_revision"):
        if not COMMIT_RE.fullmatch(str(spec[field])):
            raise ValueError(f"LLM-judge model lock {key!r} has an unpinned {field}")
    if spec["padding_side"] not in {"left", "right"} or int(spec["max_length"]) < 1:
        raise ValueError("LLM-judge padding/max_length settings are invalid")
    if not isinstance(spec["trust_remote_code"], bool) or not isinstance(
        spec["chat_template_kwargs"], dict
    ):
        raise ValueError("LLM-judge trust_remote_code/chat_template_kwargs are invalid")
    if spec["protocol_role"] not in {"frozen_primary", "smoke_test_only"}:
        raise ValueError(
            "LLM-judge protocol_role must be frozen_primary or smoke_test_only"
        )
    labels = [str(spec["negative_label"]).strip(), str(spec["positive_label"]).strip()]
    if not all(labels) or labels[0] == labels[1]:
        raise ValueError(
            "LLM-judge forced-choice labels must be non-empty and distinct"
        )
    if not str(spec["system_prompt"]).strip() or not str(spec["family"]).strip():
        raise ValueError("LLM-judge lock requires a family and frozen system_prompt")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    spec_hash = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()
    return spec, config_hash, spec_hash


def pairwise_positive_probability(
    logits, negative_token_id: int, positive_token_id: int
):
    import torch

    values = torch.as_tensor(logits)
    if values.ndim != 2:
        raise ValueError("Judge logits must be batch-by-vocabulary")
    if negative_token_id == positive_token_id:
        raise ValueError("Judge label token IDs must differ")
    pair = values[:, [negative_token_id, positive_token_id]].float()
    return torch.softmax(pair, dim=-1)[:, 1]


def contextual_label_token_ids(
    tokenizer, rendered_prompt: str, labels: list[str]
) -> list[int]:
    """Resolve labels in the exact next-token context used for scoring."""

    base_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    if not base_ids:
        raise ValueError("Rendered LLM-judge prompt tokenized to an empty sequence")
    token_ids: list[int] = []
    for label in labels:
        continued_ids = tokenizer.encode(
            rendered_prompt + str(label), add_special_tokens=False
        )
        if (
            continued_ids[: len(base_ids)] != base_ids
            or len(continued_ids) != len(base_ids) + 1
        ):
            raise ValueError(
                f"Forced-choice judge label {label!r} must append exactly one token in "
                "the registered generation context"
            )
        token_ids.append(int(continued_ids[-1]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Forced-choice judge labels map to the same contextual token")
    return token_ids


class JudgeRuntime:
    def __init__(self, spec: dict[str, Any], batch_size: int, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if batch_size < 1:
            raise ValueError("judge_batch_size must be positive")
        self.torch = torch
        self.spec = spec
        resolved_device = resolve_torch_device(device)
        # A 14B decoder leaves little unified-memory headroom on Apple
        # Silicon. Single-item scoring avoids transient padded-batch peaks;
        # cache identities are batch-size independent.
        self.batch_size = bounded_batch_size_for_device(
            batch_size,
            resolved_device,
        )
        self.resolved_device = resolved_device
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec["model_id"],
            revision=spec["tokenizer_revision"],
            padding_side=spec["padding_side"],
            trust_remote_code=spec["trust_remote_code"],
        )
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("Registered LLM judge tokenizer has no chat template")
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                raise ValueError("LLM judge tokenizer has no pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        probe_rendered = self._render("Contextual label-token validation probe.", [])
        label_ids = contextual_label_token_ids(
            self.tokenizer,
            probe_rendered,
            [str(spec["negative_label"]), str(spec["positive_label"])],
        )
        self.negative_token_id, self.positive_token_id = label_ids
        model_kwargs = {
            "revision": spec["model_revision"],
            "torch_dtype": inference_dtype_for_device(resolved_device),
            "low_cpu_mem_usage": True,
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            spec["model_id"],
            trust_remote_code=spec["trust_remote_code"],
            **model_kwargs,
        )
        self.model = self.model.to(resolved_device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model_device = next(self.model.parameters()).device
        self.model_parameter_dtype = str(next(self.model.parameters()).dtype)

    def _render(self, text: str, demonstrations: list[tuple[str, int]]) -> str:
        messages = build_judge_messages(text, demonstrations, self.spec)
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.spec["chat_template_kwargs"],
        )
        if not isinstance(rendered, str) or not rendered:
            raise ValueError("LLM judge chat template returned empty text")
        return rendered

    def score_bundle(
        self,
        bundle: dict[str, np.ndarray],
        demonstrations: list[tuple[str, int]],
    ) -> dict[str, np.ndarray]:
        rendered = [
            self._render(text, demonstrations) for text in bundle["texts"].tolist()
        ]
        scores: list[np.ndarray] = []
        token_lengths: list[int] = []
        max_length = int(self.spec["max_length"])
        for start in range(0, len(rendered), self.batch_size):
            batch_texts = rendered[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            # Truncation is disabled, so every real token is retained and the
            # attention mask sums to the exact untruncated length. This avoids a
            # redundant second full tokenization pass over each batch.
            lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
            too_long = [
                start + i for i, length in enumerate(lengths) if length > max_length
            ]
            if too_long:
                raise ValueError(
                    f"LLM-judge prompts {too_long[:5]} exceed registered max_length={max_length}; "
                    "truncation is prohibited"
                )
            encoded = {
                key: value.to(self.model_device) for key, value in encoded.items()
            }
            supports_keep = getattr(self.model, "_supports_logits_to_keep", None)
            left_padded = self.tokenizer.padding_side == "left"
            use_last_only = (
                left_padded and callable(supports_keep) and supports_keep()
            )
            forward_kwargs = {"logits_to_keep": 1} if use_last_only else {}
            with self.torch.inference_mode():
                output = self.model(**encoded, **forward_kwargs)
            if use_last_only:
                # Left padding places every real final token in the last column, so
                # the single retained logit row is exactly the forced-choice
                # next-token distribution. Requesting only that row skips the LM-head
                # projection over the whole (up to max_length) padded sequence.
                next_logits = output.logits[:, -1, :]
            else:
                mask = encoded["attention_mask"]
                positions = self.torch.arange(
                    mask.shape[1], device=mask.device
                ).unsqueeze(0)
                last_positions = (
                    positions * mask.to(dtype=self.torch.long)
                ).argmax(dim=1)
                batch_indices = self.torch.arange(mask.shape[0], device=mask.device)
                next_logits = output.logits[batch_indices, last_positions]
            probabilities = pairwise_positive_probability(
                next_logits, self.negative_token_id, self.positive_token_id
            )
            scores.append(probabilities.detach().cpu().numpy())
            token_lengths.extend(lengths)
        return {
            "labels": bundle["labels"],
            "scores": np.concatenate(scores).astype(float, copy=False),
            "example_ids": bundle["example_ids"],
            "question_ids": bundle["question_ids"],
            "prompt_sha256": np.asarray(
                [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in rendered]
            ),
            "prompt_token_lengths": np.asarray(token_lengths, dtype=np.int64),
        }


def _load_examples(task_name: str, path: str, *, reference_only: bool):
    task = TASK_REGISTRY[task_name]()
    examples = task.load(path)
    if any(
        example.metadata.get("data_origin") != "on_policy_generation"
        or example.metadata.get("generated_by_model") is not True
        for example in examples
    ):
        raise ValueError("LLM-judge baselines require on-policy model outputs")
    if reference_only:
        split_examples = {
            split: [
                example
                for example in examples
                if example.metadata.get("protocol_split") == split
            ]
            for split in ("calibration", "test")
        }
        for split, rows in split_examples.items():
            if not rows or any(example.label != 0 for example in rows):
                raise ValueError(
                    f"Reference judge {split} must use membership value 0"
                )
    else:
        split_examples = declared_protocol_split(
            examples, group_key=task.spec.grouped_split_key
        )
    return (
        split_examples,
        monitored_model_identity(examples),
        hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )


def _arrays_by_view(split_examples: dict[str, list], view: str):
    return {
        split_name: examples_to_text_arrays(examples, view)
        for split_name, examples in split_examples.items()
        if examples
    }


def _require_identity(reference: dict[str, str], other: dict[str, str]) -> None:
    if reference != other:
        raise ValueError(
            "Source, reference, and transfer judge data must come from the same "
            "monitored model and tokenizer revisions"
        )


def _nan_transfer_metrics(row: dict[str, Any]) -> None:
    for metric in (
        "auroc",
        "auprc",
        "tpr_at_reference_alert_budget",
        "tpr_at_1pct_reference_alert_budget",
        "fpr_at_frozen_threshold",
        "oracle_tpr_at_requested_fpr",
        "brier",
        "ece",
    ):
        row[f"transfer_{metric}"] = float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pinned forced-choice zero/few-shot LLM-judge baselines"
    )
    parser.add_argument("--source_task", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--source_data", required=True)
    parser.add_argument("--target_task", default=None, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--target_data", default=None)
    parser.add_argument(
        "--reference_task", required=True, choices=sorted(TASK_REGISTRY)
    )
    parser.add_argument("--reference_data", required=True)
    parser.add_argument("--judge_config", required=True)
    parser.add_argument("--judge_model_key", required=True)
    parser.add_argument("--judge_cache_dir", required=True)
    parser.add_argument("--judge_batch_size", type=int, default=8)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--views", default="prompt_text,answer_text,transcript_text")
    parser.add_argument("--modes", default="zero_shot,few_shot")
    parser.add_argument("--k_values", default="1,2,4,8,16,32")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_reference_alert_rate", type=float, default=0.01)
    parser.add_argument("--min_reference_groups", type=int, default=1000)
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto|cpu|cuda|cuda:N|mps",
    )
    parser.add_argument("--allow_dirty_code", action="store_true")
    parser.add_argument("--allow_self_judge_debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--selection_only",
        action="store_true",
        help="Score source eval and reference traffic without touching test targets.",
    )
    args = parser.parse_args()

    if bool(args.target_task) != bool(args.target_data):
        raise ValueError("--target_task and --target_data must be provided together")
    if (
        args.source_task == "reference_traffic"
        or args.target_task == "reference_traffic"
    ):
        raise ValueError(
            "reference_traffic cannot be a training or transfer task"
        )
    if args.reference_task != "reference_traffic":
        raise ValueError("Operational calibration requires reference_traffic")
    views = [value.strip() for value in args.views.split(",") if value.strip()]
    if not views or not set(views).issubset(ALLOWED_TEXT_VIEWS):
        raise ValueError(
            f"Judge views must be chosen from {sorted(ALLOWED_TEXT_VIEWS)}"
        )
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if not modes or not set(modes).issubset({"zero_shot", "few_shot"}):
        raise ValueError("Judge modes must be zero_shot and/or few_shot")
    k_values = [
        int(value.strip()) for value in args.k_values.split(",") if value.strip()
    ]
    if any(k < 1 for k in k_values) or args.seeds < 1:
        raise ValueError("Judge k values and seeds must be positive")
    target_task = args.target_task or args.source_task
    spec, config_hash, spec_hash = _load_judge_spec(
        Path(args.judge_config), args.judge_model_key
    )
    code_revision, code_dirty = _git_state()
    implementation_sha256 = _implementation_sha256()
    if code_dirty and not args.allow_dirty_code:
        raise RuntimeError(
            "Refusing LLM-judge scoring from a dirty worktree; commit the protocol or "
            "pass --allow_dirty_code for a non-confirmatory debug run"
        )

    source_examples, source_identity, source_hash = _load_examples(
        args.source_task, args.source_data, reference_only=False
    )
    if (
        spec["model_id"] == source_identity["monitored_model_id"]
        and not args.allow_self_judge_debug
    ):
        raise ValueError(
            "The registered judge must be independent of the monitored model"
        )
    reference_examples, reference_identity, reference_hash = _load_examples(
        args.reference_task, args.reference_data, reference_only=True
    )
    _require_identity(source_identity, reference_identity)
    if args.target_data and not args.selection_only:
        target_examples, target_identity, target_hash = _load_examples(
            target_task, args.target_data, reference_only=False
        )
        _require_identity(source_identity, target_identity)
    else:
        target_examples = None
        target_hash = None

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.judge_cache_dir)
    out_file = (
        results_dir
        / (
            f"{args.source_task}__to__{target_task}__llm_judge_baselines"
            f"{'__selection' if args.selection_only else ''}.jsonl"
        )
    )
    if args.overwrite and out_file.exists():
        out_file.unlink()
    existing = _load_existing_run_ids(out_file)
    predictions_dir = results_dir / "predictions"
    runtime: JudgeRuntime | None = None
    completed = 0

    with out_file.open("a", encoding="utf-8") as handle:
        for view in views:
            source = _arrays_by_view(source_examples, view)
            reference = _arrays_by_view(reference_examples, view)
            target = (
                _arrays_by_view(target_examples, view)["test"]
                if target_examples is not None
                else None
            )
            train = source["train"]

            contexts: list[tuple[str, int, int | None, FewShotSelection | None]] = []
            if "zero_shot" in modes:
                contexts.append(("zero_shot", 0, None, None))
            if "few_shot" in modes:
                for k in k_values:
                    for seed in range(args.seeds):
                        selection = sample_few_shot_train(
                            np.arange(len(train["labels"]))[:, None],
                            train["labels"],
                            k=k,
                            seed=seed,
                            balance_mode="balanced",
                            group_ids=train["question_ids"],
                            return_selection=True,
                        )
                        assert isinstance(selection, FewShotSelection)
                        contexts.append(("few_shot", k, seed, selection))

            for mode, k, context_seed, selection in contexts:
                if selection is None:
                    demo_indices = np.asarray([], dtype=np.int64)
                    demonstrations: list[tuple[str, int]] = []
                else:
                    demo_indices = selection.indices
                    demonstrations = [
                        (str(train["texts"][index]), int(train["labels"][index]))
                        for index in demo_indices
                    ]
                context_payload = {
                    "judge_spec_sha256": spec_hash,
                    "judge_config_sha256": config_hash,
                    "code_revision": code_revision,
                    "judge_implementation_sha256": implementation_sha256,
                    "view": view,
                    "mode": mode,
                    "k": k,
                    "context_seed": context_seed,
                    "source_task": args.source_task,
                    "target_task": target_task,
                    "source_data_sha256": source_hash,
                    "reference_data_sha256": reference_hash,
                    "target_data_sha256": target_hash,
                    "demonstration_example_ids": train["example_ids"][
                        demo_indices
                    ].tolist(),
                    "demonstration_labels": train["labels"][demo_indices].tolist(),
                    "demonstration_text_sha256": [
                        hashlib.sha256(
                            str(train["texts"][index]).encode("utf-8")
                        ).hexdigest()
                        for index in demo_indices
                    ],
                }
                context_hash = judge_context_hash(context_payload)
                cache_path = cache_dir / f"{context_hash}.npz"
                expected_splits = {
                    "reference_calibration",
                    "reference_holdout",
                    "source_eval",
                }
                if not args.selection_only:
                    expected_splits.add("source_test")
                if target is not None and not args.selection_only:
                    expected_splits.add("target_test")
                if cache_path.exists():
                    scored, cache_metadata = load_judge_cache(
                        cache_path,
                        expected_context_hash=context_hash,
                        expected_splits=expected_splits,
                    )
                    if cache_metadata.get("code_dirty") and not args.allow_dirty_code:
                        raise ValueError(
                            f"Judge cache {cache_path} was produced from dirty code"
                        )
                else:
                    if runtime is None:
                        runtime = JudgeRuntime(
                            spec,
                            batch_size=args.judge_batch_size,
                            device=args.device,
                        )
                    bundles = {
                        "reference_calibration": reference["calibration"],
                        "reference_holdout": reference["test"],
                        "source_eval": source["eval"],
                    }
                    if not args.selection_only:
                        bundles["source_test"] = source["test"]
                    if target is not None and not args.selection_only:
                        bundles["target_test"] = target
                    scored = {
                        split_name: runtime.score_bundle(bundle, demonstrations)
                        for split_name, bundle in bundles.items()
                    }
                    cache_metadata = {
                        "schema_version": LLM_JUDGE_CACHE_SCHEMA_VERSION,
                        "context_hash": context_hash,
                        "code_revision": code_revision,
                        "code_dirty": code_dirty,
                        "judge_implementation_sha256": implementation_sha256,
                        "judge_model_key": args.judge_model_key,
                        "judge_model_id": spec["model_id"],
                        "judge_model_revision": spec["model_revision"],
                        "judge_tokenizer_revision": spec["tokenizer_revision"],
                        "judge_family": spec["family"],
                        "judge_protocol_role": spec["protocol_role"],
                        "judge_spec_sha256": spec_hash,
                        "judge_config_sha256": config_hash,
                        "judge_requested_batch_size": int(args.judge_batch_size),
                        "judge_effective_batch_size": int(runtime.batch_size),
                        "judge_resolved_device": runtime.resolved_device,
                        "judge_model_parameter_dtype": (
                            runtime.model_parameter_dtype
                        ),
                        "score_semantics": "pairwise_softmax_forced_choice_next_token_logits",
                        "negative_label": spec["negative_label"],
                        "positive_label": spec["positive_label"],
                        "max_length": int(spec["max_length"]),
                        "monitored_model_id": source_identity["monitored_model_id"],
                        "monitored_model_revision": source_identity[
                            "monitored_model_revision"
                        ],
                        **context_payload,
                    }
                    atomic_save_judge_cache(cache_path, scored, cache_metadata)
                    scored, cache_metadata = load_judge_cache(
                        cache_path,
                        expected_context_hash=context_hash,
                        expected_splits=expected_splits,
                    )

                result_seeds = (
                    range(args.seeds) if mode == "zero_shot" else [context_seed]
                )
                for result_seed in result_seeds:
                    probe = (
                        "B3_llm_judge_zero_shot"
                        if mode == "zero_shot"
                        else "B3_llm_judge_few_shot"
                    )
                    run_id = (
                        f"{args.model}__{args.source_task}__{target_task}__{probe}__"
                        f"{args.judge_model_key}-{spec_hash[:12]}__layer-3__{view}__"
                        f"k{k}__seed{result_seed}__"
                        f"{'none' if mode == 'zero_shot' else 'balanced'}"
                        f"{'__selection' if args.selection_only else ''}"
                    )
                    prediction_path = _prediction_path(predictions_dir, run_id)
                    if run_id in existing and prediction_path.exists():
                        continue
                    n_reference_groups = (
                        require_independent_reference_groups(
                            scored["reference_calibration"]["question_ids"],
                            min_reference_groups=args.min_reference_groups,
                        )
                    )
                    n_reference_holdout_groups = require_independent_reference_groups(
                        scored["reference_holdout"]["question_ids"],
                        min_reference_groups=args.min_reference_groups,
                    )
                    require_disjoint_reference_groups(
                        scored["reference_calibration"]["question_ids"],
                        scored["reference_holdout"]["question_ids"],
                    )
                    threshold = select_threshold_at_alert_rate(
                        scored["reference_calibration"]["scores"],
                        max_alert_rate=args.max_reference_alert_rate,
                        min_reference=args.min_reference_groups,
                    )
                    holdout_alerts = alert_rate_summary(
                        scored["reference_holdout"]["scores"], threshold
                    )
                    row: dict[str, Any] = {
                        "status": "ok",
                        "error": False,
                        "run_id": run_id,
                        "execution_mode": (
                            "selection"
                            if args.selection_only
                            else "confirmatory"
                        ),
                        "probe": probe,
                        "method_family": "black_box_llm_judge",
                        "judge_mode": mode,
                        "k": k,
                        "k_unit": (
                            "no_labeled_examples"
                            if mode == "zero_shot"
                            else "positive_scenario_groups"
                        ),
                        "seed": int(result_seed),
                        "seed_role": (
                            "paired_replication_only"
                            if mode == "zero_shot"
                            else "few_shot_demonstration_sampling"
                        ),
                        "balance_mode": "none" if mode == "zero_shot" else "balanced",
                        "model": args.model,
                        **source_identity,
                        "layer": -3,
                        "view": view,
                        "source_task": args.source_task,
                        "target_task": target_task,
                        "judge_model_key": args.judge_model_key,
                        "judge_model_id": spec["model_id"],
                        "judge_model_revision": spec["model_revision"],
                        "judge_protocol_role": spec["protocol_role"],
                        "judge_spec_sha256": spec_hash,
                        "code_revision": code_revision,
                        "code_dirty": code_dirty,
                        "judge_implementation_sha256": implementation_sha256,
                        "judge_score_semantics": cache_metadata["score_semantics"],
                        "judge_cache_file": cache_metadata["cache_file"],
                        "judge_cache_sha256": cache_metadata["cache_sha256"],
                        "judge_effective_batch_size": cache_metadata.get(
                            "judge_effective_batch_size"
                        ),
                        "judge_resolved_device": cache_metadata.get(
                            "judge_resolved_device"
                        ),
                        "judge_model_parameter_dtype": cache_metadata.get(
                            "judge_model_parameter_dtype"
                        ),
                        "operating_threshold": float(threshold),
                        "requested_max_reference_alert_rate": args.max_reference_alert_rate,
                        "threshold_source": "unlabeled_reference_traffic",
                        "calibration_method": "split_conformal_upper_tail_v1",
                        "n_reference_calibration": int(
                            len(scored["reference_calibration"]["scores"])
                        ),
                        "n_reference_groups": n_reference_groups,
                        "n_reference_holdout": int(
                            len(scored["reference_holdout"]["scores"])
                        ),
                        "n_reference_holdout_groups": n_reference_holdout_groups,
                        "reference_calibration_alert_rate": compute_alert_rate(
                            scored["reference_calibration"]["scores"], threshold
                        ),
                        "reference_holdout_alert_count": holdout_alerts["alerts"],
                        "reference_holdout_alert_rate": holdout_alerts["rate"],
                        "reference_holdout_alert_rate_ci_low": holdout_alerts["ci_low"],
                        "reference_holdout_alert_rate_ci_high": holdout_alerts["ci_high"],
                        "reference_holdout_alert_budget_violation": bool(
                            holdout_alerts["ci_low"]
                            > args.max_reference_alert_rate
                        ),
                        "conformal_p_value_resolution": 1.0
                        / (len(scored["reference_calibration"]["scores"]) + 1.0),
                        "n_train_pos": int(k if mode == "few_shot" else 0),
                        "n_train_neg": int(k if mode == "few_shot" else 0),
                        "n_train_groups": int(k if mode == "few_shot" else 0),
                        "scores_are_probabilities": True,
                        "train_example_ids": train["example_ids"][
                            demo_indices
                        ].tolist(),
                        "train_question_ids": train["question_ids"][
                            demo_indices
                        ].tolist(),
                        "max_judge_prompt_tokens": int(
                            max(
                                np.max(split["prompt_token_lengths"])
                                for split in scored.values()
                            )
                        ),
                    }
                    for prefix, split_name in (("eval", "source_eval"),):
                        row.update(
                            _metric_payload(
                                prefix,
                                scored[split_name],
                                threshold,
                                probability_scores=True,
                                max_alert_rate=args.max_reference_alert_rate,
                            )
                        )
                    if "source_test" in scored:
                        row.update(
                            _metric_payload(
                                "test",
                                scored["source_test"],
                                threshold,
                                probability_scores=True,
                                max_alert_rate=args.max_reference_alert_rate,
                            )
                        )
                    else:
                        for metric in (
                            "auroc",
                            "auprc",
                            "tpr_at_reference_alert_budget",
                            "tpr_at_1pct_reference_alert_budget",
                            "fpr_at_frozen_threshold",
                            "oracle_tpr_at_requested_fpr",
                            "brier",
                            "ece",
                        ):
                            row[f"test_{metric}"] = float("nan")
                    if "target_test" in scored:
                        row.update(
                            _metric_payload(
                                "transfer",
                                scored["target_test"],
                                threshold,
                                probability_scores=True,
                                max_alert_rate=args.max_reference_alert_rate,
                            )
                        )
                    else:
                        _nan_transfer_metrics(row)
                    prediction_rows = [
                        record
                        for split_name, split_scores in scored.items()
                        for record in _prediction_records(
                            run_id, split_name, split_scores, threshold
                        )
                    ]
                    _atomic_write_jsonl(prediction_path, prediction_rows)
                    row["prediction_file"] = str(prediction_path)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    existing.add(run_id)
                    completed += 1

    print(f"completed {completed} LLM-judge runs; saved {out_file}")


if __name__ == "__main__":
    main()
