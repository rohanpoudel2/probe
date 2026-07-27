from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

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
from data.generation_confidence import (
    CONFIDENCE_FEATURE_NAMES,
    GENERATION_CONFIDENCE_SCHEMA_VERSION,
    confidence_feature_vector,
)
from data.group_splitting import declared_protocol_split
from data.rollout_schema import content_hash
from data.text_views import monitored_model_identity
from evaluation.metrics import (
    alert_rate_summary,
    compute_alert_rate,
    require_disjoint_reference_groups,
    require_independent_reference_groups,
    select_threshold_at_alert_rate,
)
from task_benchmark.sampling import FewShotSelection, sample_few_shot_train
from tasks import TASK_REGISTRY


def _arrays(examples) -> dict[str, np.ndarray]:
    rows = list(examples)
    if not rows:
        raise ValueError("Output-confidence baseline received an empty dataset")
    features = np.vstack(
        [
            confidence_feature_vector(example.metadata.get("generation"))
            for example in rows
        ]
    )
    return {
        "features": features,
        "labels": np.asarray([example.label for example in rows], dtype=np.int64),
        "example_ids": np.asarray([example.example_id for example in rows], dtype=str),
        "question_ids": np.asarray(
            [example.question_id or example.example_id for example in rows], dtype=str
        ),
    }


def _load_splits(task_name: str, path: str):
    task = TASK_REGISTRY[task_name]()
    examples = task.load(path)
    if any(
        example.metadata.get("data_origin") != "on_policy_generation"
        or example.metadata.get("generated_by_model") is not True
        for example in examples
    ):
        raise ValueError("Output-confidence baselines require on-policy model outputs")
    splits = declared_protocol_split(examples, group_key=task.spec.grouped_split_key)
    return (
        {name: _arrays(rows) for name, rows in splits.items() if rows},
        monitored_model_identity(examples),
        hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )


def _load_reference_partitions(task_name: str, path: str):
    task = TASK_REGISTRY[task_name]()
    examples = task.load(path)
    partitions = {
        split: [
            example
            for example in examples
            if example.metadata.get("protocol_split") == split
        ]
        for split in ("calibration", "test")
    }
    for split, rows in partitions.items():
        if not rows or any(example.label != 0 for example in rows):
            raise ValueError(
                f"Reference confidence {split} must use membership value 0"
            )
    return (
        {split: _arrays(rows) for split, rows in partitions.items()},
        monitored_model_identity(examples),
        hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )


def _score_split(
    scaler, classifier, bundle: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    scores = classifier.predict_proba(scaler.transform(bundle["features"]))[:, 1]
    return {
        "labels": bundle["labels"],
        "scores": np.asarray(scores, dtype=float),
        "example_ids": bundle["example_ids"],
        "question_ids": bundle["question_ids"],
    }


def _require_identity(reference: dict[str, str], other: dict[str, str]) -> None:
    if reference != other:
        raise ValueError(
            "Source, reference, and transfer confidence data must come from the same "
            "monitored model and tokenizer revisions"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched baselines on generation-time output-confidence traces"
    )
    parser.add_argument("--source_task", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--source_data", required=True)
    parser.add_argument("--target_task", default=None, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--target_data", default=None)
    parser.add_argument(
        "--reference_task", required=True, choices=sorted(TASK_REGISTRY)
    )
    parser.add_argument("--reference_data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--k_values", default="1,2,4,8,16,32")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_reference_alert_rate", type=float, default=0.01)
    parser.add_argument("--min_reference_groups", type=int, default=1000)
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
    for role, task_name in (
        ("source", args.source_task),
        ("target", args.target_task),
    ):
        if task_name is None:
            continue
        unavailable = getattr(
            TASK_REGISTRY[task_name]().spec, "unavailable_baselines", {}
        )
        reason = unavailable.get("B4_output_confidence_logistic")
        if reason:
            raise ValueError(
                f"B4 output-confidence baseline is unavailable for {role} task "
                f"{task_name}: {reason}"
            )
    k_values = [
        int(value.strip()) for value in args.k_values.split(",") if value.strip()
    ]
    if not k_values or any(k < 1 for k in k_values) or args.seeds < 1:
        raise ValueError("k values and seeds must be positive")
    target_task = args.target_task or args.source_task

    source, source_identity, source_hash = _load_splits(
        args.source_task, args.source_data
    )
    reference, reference_identity, reference_hash = _load_reference_partitions(
        args.reference_task, args.reference_data
    )
    _require_identity(source_identity, reference_identity)
    if args.target_data and not args.selection_only:
        target_splits, target_identity, target_hash = _load_splits(
            target_task, args.target_data
        )
        _require_identity(source_identity, target_identity)
        target = target_splits["test"]
    else:
        target = None
        target_hash = None

    feature_spec_sha256 = content_hash(
        {
            "schema_version": GENERATION_CONFIDENCE_SCHEMA_VERSION,
            "feature_names": CONFIDENCE_FEATURE_NAMES,
        }
    )
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = (
        results_dir
        / (
            f"{args.source_task}__to__{target_task}__output_confidence_baselines"
            f"{'__selection' if args.selection_only else ''}.jsonl"
        )
    )
    if args.overwrite and out_file.exists():
        out_file.unlink()
    existing = _load_existing_run_ids(out_file)
    predictions_dir = results_dir / "predictions"
    completed = 0

    with out_file.open("a", encoding="utf-8") as handle:
        train = source["train"]
        for k in k_values:
            for seed in range(args.seeds):
                run_id = (
                    f"{args.model}__{args.source_task}__{target_task}__"
                    f"B4_output_confidence_logistic__layer-4__generation_confidence__"
                    f"k{k}__seed{seed}__balanced"
                    f"{'__selection' if args.selection_only else ''}"
                )
                prediction_path = _prediction_path(predictions_dir, run_id)
                if run_id in existing and prediction_path.exists():
                    continue
                selection = sample_few_shot_train(
                    train["features"],
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
                    "reference_calibration": _score_split(
                        scaler, classifier, reference["calibration"]
                    ),
                    "reference_holdout": _score_split(
                        scaler, classifier, reference["test"]
                    ),
                    "source_eval": _score_split(scaler, classifier, source["eval"]),
                }
                if not args.selection_only:
                    scored["source_test"] = _score_split(
                        scaler, classifier, source["test"]
                    )
                if target is not None and not args.selection_only:
                    scored["target_test"] = _score_split(scaler, classifier, target)
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
                row = {
                    "status": "ok",
                    "error": False,
                    "run_id": run_id,
                    "execution_mode": (
                        "selection"
                        if args.selection_only
                        else "confirmatory"
                    ),
                    "probe": "B4_output_confidence_logistic",
                    "method_family": "black_box_output_confidence",
                    "k": k,
                    "k_unit": "positive_scenario_groups",
                    "seed": seed,
                    "balance_mode": "balanced",
                    "model": args.model,
                    **source_identity,
                    "layer": -4,
                    "view": "generation_confidence",
                    "source_task": args.source_task,
                    "target_task": target_task,
                    "generation_confidence_schema": GENERATION_CONFIDENCE_SCHEMA_VERSION,
                    "confidence_feature_names": list(CONFIDENCE_FEATURE_NAMES),
                    "confidence_feature_spec_sha256": feature_spec_sha256,
                    "source_data_sha256": source_hash,
                    "reference_data_sha256": reference_hash,
                    "target_data_sha256": target_hash,
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
                        holdout_alerts["ci_low"] > args.max_reference_alert_rate
                    ),
                    "conformal_p_value_resolution": 1.0
                    / (len(scored["reference_calibration"]["scores"]) + 1.0),
                    "n_train_pos": int(np.sum(selection.y == 1)),
                    "n_train_neg": int(np.sum(selection.y == 0)),
                    "n_train_groups": int(len(np.unique(selection.group_ids))),
                    "scores_are_probabilities": True,
                    "classifier": "standard_scaler_plus_logistic_C1_liblinear",
                    "train_example_ids": train["example_ids"][
                        selection.indices
                    ].tolist(),
                    "train_question_ids": selection.group_ids.tolist(),
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

    print(f"completed {completed} output-confidence runs; saved {out_file}")


if __name__ == "__main__":
    main()
