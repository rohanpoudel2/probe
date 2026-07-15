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
    require_independent_calibration_negatives,
    select_threshold_at_fpr,
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


def _load_calibration(task_name: str, path: str):
    task = TASK_REGISTRY[task_name]()
    examples = task.load(path)
    if any(example.label != 0 for example in examples):
        raise ValueError("Dedicated benign confidence calibration must be all-negative")
    return (
        _arrays(examples),
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
            "Source, calibration, and transfer confidence data must come from the same "
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
        "--calibration_task", default=None, choices=sorted(TASK_REGISTRY)
    )
    parser.add_argument("--calibration_data", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--k_values", default="1,2,4,8,16,32")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_fpr", type=float, default=0.01)
    parser.add_argument("--min_calibration_negatives", type=int, default=1000)
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
    if args.calibration_data:
        calibration, calibration_identity, calibration_hash = _load_calibration(
            args.calibration_task, args.calibration_data
        )
        _require_identity(source_identity, calibration_identity)
    else:
        if "calibration" not in source:
            raise ValueError(
                "Source data has no calibration split; provide dedicated benign calibration"
            )
        calibration = source["calibration"]
        calibration_hash = source_hash
    if args.target_data:
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
        / f"{args.source_task}__to__{target_task}__output_confidence_baselines.jsonl"
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
                    "source_calibration": _score_split(scaler, classifier, calibration),
                    "source_eval": _score_split(scaler, classifier, source["eval"]),
                    "source_test": _score_split(scaler, classifier, source["test"]),
                }
                if target is not None:
                    scored["target_test"] = _score_split(scaler, classifier, target)
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
                row = {
                    "status": "ok",
                    "error": False,
                    "run_id": run_id,
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
                    "calibration_data_sha256": calibration_hash,
                    "target_data_sha256": target_hash,
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
                    "train_example_ids": train["example_ids"][
                        selection.indices
                    ].tolist(),
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

    print(f"completed {completed} output-confidence runs; saved {out_file}")


if __name__ == "__main__":
    main()
