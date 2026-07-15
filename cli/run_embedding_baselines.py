from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cli.run_task_sweep import (
    _atomic_write_jsonl,
    _load_existing_run_ids,
    _metric_payload,
    _prediction_path,
    _prediction_records,
)
from data.text_embedding_cache import load_text_embedding_cache
from data.text_views import ALLOWED_TEXT_VIEWS
from evaluation.metrics import (
    require_independent_calibration_negatives,
    select_threshold_at_fpr,
)
from task_benchmark.sampling import FewShotSelection, sample_few_shot_train
from tasks import TASK_REGISTRY


def _cache_path(directory: str, task_name: str, view: str) -> Path:
    return Path(directory) / f"{task_name}__{view}.npz"


def _subset(cache: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    mask = cache["protocol_splits"] == split
    if not np.any(mask):
        raise ValueError(f"Cache {cache['cache_file']} has no {split} rows")
    return {
        "embeddings": cache["embeddings"][mask],
        "labels": cache["labels"][mask],
        "example_ids": cache["example_ids"][mask],
        "question_ids": cache["question_ids"][mask],
    }


def _validate_cache(
    cache: dict[str, Any],
    *,
    task_name: str,
    view: str,
    allow_truncated: bool,
) -> None:
    metadata = cache["metadata"]
    if metadata["task_name"] != task_name or metadata["view"] != view:
        raise ValueError(
            f"Cache {cache['cache_file']} is {metadata['task_name']} / {metadata['view']}, "
            f"not {task_name} / {view}"
        )
    if np.any(cache["truncated"]) and not allow_truncated:
        raise ValueError(
            f"Cache {cache['cache_file']} contains truncated text; use a larger registered "
            "max_length or explicitly pass --allow_truncated_cache for a pilot"
        )


def _assert_compatible(reference: dict[str, Any], other: dict[str, Any]) -> None:
    reference_meta = reference["metadata"]
    other_meta = other["metadata"]
    for key in (
        "embedding_model_id",
        "embedding_model_revision",
        "embedding_tokenizer_revision",
        "embedding_spec_sha256",
        "embedding_config_sha256",
        "pooling",
        "padding_side",
        "normalized",
        "max_length",
        "instruction",
        "instruction_format",
        "monitored_model_id",
        "monitored_model_revision",
        "monitored_tokenizer_revision",
        "code_revision",
    ):
        if reference_meta[key] != other_meta[key]:
            raise ValueError(
                f"Incompatible text embedding caches: {key} differs between "
                f"{reference['cache_file']} and {other['cache_file']}"
            )
    if reference["embeddings"].shape[1] != other["embeddings"].shape[1]:
        raise ValueError("Text embedding cache dimensions differ")


def _score_split(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    bundle: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    scores = classifier.predict_proba(scaler.transform(bundle["embeddings"]))[:, 1]
    return {
        "labels": bundle["labels"],
        "scores": np.asarray(scores, dtype=float),
        "example_ids": bundle["example_ids"],
        "question_ids": bundle["question_ids"],
    }


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched few-shot logistic baselines on reusable text embeddings"
    )
    parser.add_argument("--source_task", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--source_cache_dir", required=True)
    parser.add_argument("--target_task", default=None, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--target_cache_dir", default=None)
    parser.add_argument("--calibration_task", default=None, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--calibration_cache_dir", default=None)
    parser.add_argument("--model", required=True, help="Registered display name of the monitored model")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--views", default="prompt_text,answer_text,transcript_text")
    parser.add_argument("--k_values", default="1,2,4,8,16,32")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_fpr", type=float, default=0.01)
    parser.add_argument("--min_calibration_negatives", type=int, default=1000)
    parser.add_argument("--allow_dirty_cache", action="store_true")
    parser.add_argument("--allow_truncated_cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if bool(args.target_task) != bool(args.target_cache_dir):
        raise ValueError("--target_task and --target_cache_dir must be provided together")
    if bool(args.calibration_task) != bool(args.calibration_cache_dir):
        raise ValueError(
            "--calibration_task and --calibration_cache_dir must be provided together"
        )
    if args.source_task == "benign_calibration":
        raise ValueError("benign_calibration cannot be used as a few-shot source task")
    if args.target_task == "benign_calibration":
        raise ValueError("benign_calibration cannot be used as a transfer target")
    if args.calibration_task and args.calibration_task != "benign_calibration":
        raise ValueError("Dedicated calibration must use the benign_calibration task contract")
    views = [value.strip() for value in args.views.split(",") if value.strip()]
    if not views or not set(views).issubset(ALLOWED_TEXT_VIEWS):
        raise ValueError(f"Text views must be chosen from {sorted(ALLOWED_TEXT_VIEWS)}")
    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    if any(value < 1 for value in k_values) or args.seeds < 1:
        raise ValueError("k values and seeds must be positive")
    target_task = args.target_task or args.source_task

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = (
        results_dir
        / f"{args.source_task}__to__{target_task}__text_embedding_baselines.jsonl"
    )
    if args.overwrite and out_file.exists():
        out_file.unlink()
    existing = _load_existing_run_ids(out_file)
    predictions_dir = results_dir / "predictions"
    completed = 0

    with out_file.open("a", encoding="utf-8") as handle:
        for view in views:
            source = load_text_embedding_cache(
                _cache_path(args.source_cache_dir, args.source_task, view),
                require_clean_code=not args.allow_dirty_cache,
            )
            _validate_cache(
                source,
                task_name=args.source_task,
                view=view,
                allow_truncated=args.allow_truncated_cache,
            )
            calibration_cache = (
                load_text_embedding_cache(
                    _cache_path(args.calibration_cache_dir, args.calibration_task, view),
                    require_clean_code=not args.allow_dirty_cache,
                )
                if args.calibration_cache_dir
                else source
            )
            _validate_cache(
                calibration_cache,
                task_name=args.calibration_task or args.source_task,
                view=view,
                allow_truncated=args.allow_truncated_cache,
            )
            _assert_compatible(source, calibration_cache)
            target_cache = (
                load_text_embedding_cache(
                    _cache_path(args.target_cache_dir, args.target_task, view),
                    require_clean_code=not args.allow_dirty_cache,
                )
                if args.target_cache_dir
                else None
            )
            if target_cache is not None:
                _validate_cache(
                    target_cache,
                    task_name=args.target_task,
                    view=view,
                    allow_truncated=args.allow_truncated_cache,
                )
                _assert_compatible(source, target_cache)

            train = _subset(source, "train")
            source_eval = _subset(source, "eval")
            source_test = _subset(source, "test")
            calibration = _subset(calibration_cache, "calibration")
            if args.calibration_cache_dir and np.any(calibration["labels"] != 0):
                raise ValueError("Dedicated benign calibration cache must be all-negative")
            target_test = _subset(target_cache, "test") if target_cache is not None else None
            meta = source["metadata"]
            model_slug = _slug(args.model)
            encoder_slug = _slug(str(meta["embedding_model_id"]))
            spec_short = str(meta["embedding_spec_sha256"])[:12]

            for k in k_values:
                for seed in range(args.seeds):
                    run_id = (
                        f"{model_slug}__{args.source_task}__{target_task}__B2_text_embedding"
                        f"__{encoder_slug}-{spec_short}__layer-2__{view}__k{k}__seed{seed}__balanced"
                    )
                    prediction_path = _prediction_path(predictions_dir, run_id)
                    if run_id in existing and prediction_path.exists():
                        continue
                    selection = sample_few_shot_train(
                        train["embeddings"],
                        train["labels"],
                        k=k,
                        seed=seed,
                        balance_mode="balanced",
                        group_ids=train["question_ids"],
                        return_selection=True,
                    )
                    assert isinstance(selection, FewShotSelection)
                    scaler = StandardScaler()
                    selected_scaled = scaler.fit_transform(selection.X)
                    classifier = LogisticRegression(
                        C=1.0,
                        solver="liblinear",
                        max_iter=1000,
                        random_state=seed,
                    )
                    classifier.fit(selected_scaled, selection.y)
                    scored = {
                        "source_calibration": _score_split(
                            scaler, classifier, calibration
                        ),
                        "source_eval": _score_split(scaler, classifier, source_eval),
                        "source_test": _score_split(scaler, classifier, source_test),
                    }
                    if target_test is not None:
                        scored["target_test"] = _score_split(
                            scaler, classifier, target_test
                        )
                    n_calibration_negative_groups = require_independent_calibration_negatives(
                        scored["source_calibration"]["labels"],
                        scored["source_calibration"]["question_ids"],
                        min_negative_groups=args.min_calibration_negatives,
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
                        "probe": "B2_text_embedding_logistic",
                        "method_family": "black_box_text_embedding",
                        "k": k,
                        "k_unit": "positive_scenario_groups",
                        "seed": seed,
                        "balance_mode": "balanced",
                        "model": args.model,
                        "monitored_model_id": meta["monitored_model_id"],
                        "model_revision": meta["monitored_model_revision"],
                        "layer": -2,
                        "view": view,
                        "source_task": args.source_task,
                        "target_task": target_task,
                        "embedding_model": meta["embedding_model_id"],
                        "embedding_model_revision": meta["embedding_model_revision"],
                        "embedding_spec_sha256": meta["embedding_spec_sha256"],
                        "source_cache_sha256": source["cache_sha256"],
                        "calibration_cache_sha256": calibration_cache["cache_sha256"],
                        "target_cache_sha256": (
                            target_cache["cache_sha256"] if target_cache is not None else None
                        ),
                        "operating_threshold": float(threshold),
                        "requested_max_fpr": args.max_fpr,
                        "threshold_source": "source_calibration_negatives",
                        "n_calibration_negative": int(
                            np.sum(scored["source_calibration"]["labels"] == 0)
                        ),
                        "n_calibration_negative_groups": n_calibration_negative_groups,
                        "n_train_pos": int(np.sum(selection.y == 1)),
                        "n_train_neg": int(np.sum(selection.y == 0)),
                        "n_train_groups": int(len(np.unique(selection.group_ids))),
                        "scores_are_probabilities": True,
                        "classifier": "standard_scaler_plus_logistic_C1_liblinear",
                        "train_example_ids": train["example_ids"][selection.indices].tolist(),
                        "train_question_ids": selection.group_ids.tolist(),
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

    print(f"completed {completed} embedding-monitor runs; saved {out_file}")


if __name__ == "__main__":
    main()
