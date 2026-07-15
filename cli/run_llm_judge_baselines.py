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
    require_independent_calibration_negatives,
    select_threshold_at_fpr,
)
from task_benchmark.sampling import FewShotSelection, sample_few_shot_train
from tasks import TASK_REGISTRY
from cli.common import resolve_torch_device


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
    if spec["protocol_role"] not in {"frozen_primary", "pilot_only"}:
        raise ValueError("LLM-judge protocol_role must be frozen_primary or pilot_only")
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
        self.batch_size = batch_size
        resolved_device = resolve_torch_device(device)
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
            "torch_dtype": "auto",
        }
        if resolved_device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            spec["model_id"],
            trust_remote_code=spec["trust_remote_code"],
            **model_kwargs,
        )
        if resolved_device != "auto":
            self.model = self.model.to(resolved_device)
        self.model.eval()
        self.model_device = next(self.model.parameters()).device

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
            untruncated = self.tokenizer(
                batch_texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            lengths = [len(ids) for ids in untruncated]
            too_long = [
                start + i for i, length in enumerate(lengths) if length > max_length
            ]
            if too_long:
                raise ValueError(
                    f"LLM-judge prompts {too_long[:5]} exceed registered max_length={max_length}; "
                    "truncation is prohibited"
                )
            encoded = self.tokenizer(
                batch_texts,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.model_device) for key, value in encoded.items()
            }
            with self.torch.inference_mode():
                output = self.model(**encoded)
            mask = encoded["attention_mask"]
            positions = self.torch.arange(mask.shape[1], device=mask.device).unsqueeze(
                0
            )
            last_positions = (positions * mask.to(dtype=self.torch.long)).argmax(dim=1)
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


def _load_examples(task_name: str, path: str, *, calibration_only: bool):
    task = TASK_REGISTRY[task_name]()
    examples = task.load(path)
    if any(
        example.metadata.get("data_origin") != "on_policy_generation"
        or example.metadata.get("generated_by_model") is not True
        for example in examples
    ):
        raise ValueError("LLM-judge baselines require on-policy model outputs")
    if calibration_only:
        if any(example.label != 0 for example in examples):
            raise ValueError("Dedicated benign judge calibration must be all-negative")
        split_examples = {"calibration": examples}
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
            "Source, calibration, and transfer judge data must come from the same "
            "monitored model and tokenizer revisions"
        )


def _nan_transfer_metrics(row: dict[str, Any]) -> None:
    for metric in (
        "auroc",
        "auprc",
        "recall_at_frozen_fpr",
        "recall_at_1pct_fpr",
        "fpr_at_frozen_threshold",
        "oracle_recall_at_requested_fpr",
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
        "--calibration_task", default=None, choices=sorted(TASK_REGISTRY)
    )
    parser.add_argument("--calibration_data", default=None)
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
    parser.add_argument("--max_fpr", type=float, default=0.01)
    parser.add_argument("--min_calibration_negatives", type=int, default=1000)
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto|cpu|cuda|cuda:N|mps",
    )
    parser.add_argument("--allow_dirty_code", action="store_true")
    parser.add_argument("--allow_self_judge_pilot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if bool(args.target_task) != bool(args.target_data):
        raise ValueError("--target_task and --target_data must be provided together")
    if bool(args.calibration_task) != bool(args.calibration_data):
        raise ValueError(
            "--calibration_task and --calibration_data must be provided together"
        )
    if (
        args.source_task == "benign_calibration"
        or args.target_task == "benign_calibration"
    ):
        raise ValueError(
            "benign_calibration can only be supplied through --calibration_task"
        )
    if args.calibration_task and args.calibration_task != "benign_calibration":
        raise ValueError(
            "Dedicated calibration must use the benign_calibration task contract"
        )
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
            "pass --allow_dirty_code for a non-final pilot"
        )

    source_examples, source_identity, source_hash = _load_examples(
        args.source_task, args.source_data, calibration_only=False
    )
    if (
        spec["model_id"] == source_identity["monitored_model_id"]
        and not args.allow_self_judge_pilot
    ):
        raise ValueError(
            "The registered judge must be independent of the monitored model"
        )
    if args.calibration_data:
        calibration_examples, calibration_identity, calibration_hash = _load_examples(
            args.calibration_task, args.calibration_data, calibration_only=True
        )
        _require_identity(source_identity, calibration_identity)
    else:
        if not source_examples.get("calibration"):
            raise ValueError(
                "Source data has no calibration split; provide dedicated benign calibration"
            )
        calibration_examples = {"calibration": source_examples["calibration"]}
        calibration_hash = source_hash
    if args.target_data:
        target_examples, target_identity, target_hash = _load_examples(
            target_task, args.target_data, calibration_only=False
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
        / f"{args.source_task}__to__{target_task}__llm_judge_baselines.jsonl"
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
            calibration = _arrays_by_view(calibration_examples, view)["calibration"]
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
                    "calibration_data_sha256": calibration_hash,
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
                    "source_calibration",
                    "source_eval",
                    "source_test",
                }
                if target is not None:
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
                        "source_calibration": calibration,
                        "source_eval": source["eval"],
                        "source_test": source["test"],
                    }
                    if target is not None:
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
                    )
                    prediction_path = _prediction_path(predictions_dir, run_id)
                    if run_id in existing and prediction_path.exists():
                        continue
                    n_calibration_negative_groups = (
                        require_independent_calibration_negatives(
                            scored["source_calibration"]["labels"],
                            scored["source_calibration"]["question_ids"],
                            min_negative_groups=args.min_calibration_negatives,
                        )
                    )
                    threshold = select_threshold_at_fpr(
                        scored["source_calibration"]["labels"],
                        scored["source_calibration"]["scores"],
                        max_fpr=args.max_fpr,
                        min_negatives=args.min_calibration_negatives,
                    )
                    row: dict[str, Any] = {
                        "status": "ok",
                        "error": False,
                        "run_id": run_id,
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
                        "operating_threshold": float(threshold),
                        "requested_max_fpr": args.max_fpr,
                        "threshold_source": "source_calibration_negatives",
                        "n_calibration_negative": int(
                            np.sum(scored["source_calibration"]["labels"] == 0)
                        ),
                        "n_calibration_negative_groups": n_calibration_negative_groups,
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
                    for prefix, split_name in (
                        ("calibration", "source_calibration"),
                        ("eval", "source_eval"),
                        ("test", "source_test"),
                    ):
                        row.update(
                            _metric_payload(
                                prefix,
                                scored[split_name],
                                threshold,
                                probability_scores=True,
                                max_fpr=args.max_fpr,
                            )
                        )
                    if "target_test" in scored:
                        row.update(
                            _metric_payload(
                                "transfer",
                                scored["target_test"],
                                threshold,
                                probability_scores=True,
                                max_fpr=args.max_fpr,
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
