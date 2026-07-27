from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from cli.run_task_sweep import (
    _atomic_write_jsonl,
    _load_existing_run_ids,
    _metric_payload,
    _prediction_path,
    _prediction_records,
)
from data.group_splitting import declared_protocol_split
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


def _score_text_split(vectorizer, classifier, bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scores = classifier.predict_proba(vectorizer.transform(bundle["texts"]))[:, 1]
    return {
        "labels": bundle["labels"],
        "scores": np.asarray(scores, dtype=float),
        "example_ids": bundle["example_ids"],
        "question_ids": bundle["question_ids"],
    }


def _load_splits(task_name: str, path: str, view: str):
    task = TASK_REGISTRY[task_name]()
    examples = task.load(path)
    invalid = [
        example.example_id
        for example in examples
        if example.metadata.get("data_origin") != "on_policy_generation"
    ]
    if invalid:
        raise ValueError(
            f"Text baselines require on-policy model outputs; invalid examples include {invalid[:5]}"
        )
    splits = declared_protocol_split(examples, group_key=task.spec.grouped_split_key)
    # Skip splits with no examples (a task using dedicated reference traffic may
    # have an empty task-internal calibration split); building a text view
    # from an empty split would fail and such splits are never consumed here.
    return (
        {
            name: examples_to_text_arrays(rows, view)
            for name, rows in splits.items()
            if rows
        },
        monitored_model_identity(examples),
    )


def _load_reference_partitions(
    task_name: str, path: str, view: str
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    task = TASK_REGISTRY[task_name]()
    examples = task.load(path)
    if any(example.metadata.get("data_origin") != "on_policy_generation" for example in examples):
        raise ValueError("Reference traffic must contain only on-policy outputs")
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
                f"Reference {split} must use membership value 0 and contain rows"
            )
    return (
        {
            split: examples_to_text_arrays(rows, view)
            for split, rows in partitions.items()
        },
        monitored_model_identity(examples),
    )


def _require_same_monitored_model(reference: dict[str, str], other: dict[str, str]) -> None:
    if reference != other:
        raise ValueError(
            "Source, reference, and transfer text data must come from the same "
            "monitored model and tokenizer revisions"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched TF-IDF black-box monitor baselines")
    parser.add_argument("--source_task", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--source_data", required=True)
    parser.add_argument("--target_task", default=None, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--target_data", default=None)
    parser.add_argument("--reference_task", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--reference_data", required=True)
    parser.add_argument("--model", required=True, help="Model that generated the monitored rollouts")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--views", default="prompt_text,answer_text,transcript_text")
    parser.add_argument("--k_values", default="1,2,4,8,16,32")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_reference_alert_rate", type=float, default=0.01)
    parser.add_argument("--min_reference_groups", type=int, default=1000)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--selection_only",
        action="store_true",
        help="Score source eval and reference traffic without touching test targets.",
    )
    args = parser.parse_args()

    if bool(args.target_task) != bool(args.target_data):
        raise ValueError("--target_task and --target_data must be provided together")
    if args.source_task == "reference_traffic" or args.target_task == "reference_traffic":
        raise ValueError("reference_traffic cannot be a training or transfer task")
    if args.reference_task != "reference_traffic":
        raise ValueError("Operational calibration requires reference_traffic")
    views = [value.strip() for value in args.views.split(",") if value.strip()]
    if not set(views).issubset(ALLOWED_TEXT_VIEWS):
        raise ValueError(f"Text views must be chosen from {sorted(ALLOWED_TEXT_VIEWS)}")
    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    target_task = args.target_task or args.source_task

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stage_suffix = "__selection" if args.selection_only else ""
    out_file = (
        results_dir
        / f"{args.source_task}__to__{target_task}__text_baselines{stage_suffix}.jsonl"
    )
    if args.overwrite and out_file.exists():
        out_file.unlink()
    existing = _load_existing_run_ids(out_file)
    predictions_dir = results_dir / "predictions"
    completed = 0

    with out_file.open("a", encoding="utf-8") as handle:
        for view in views:
            source, source_identity = _load_splits(args.source_task, args.source_data, view)
            reference, reference_identity = _load_reference_partitions(
                args.reference_task, args.reference_data, view
            )
            _require_same_monitored_model(source_identity, reference_identity)
            if args.target_data and not args.selection_only:
                target_splits, target_identity = _load_splits(
                    target_task, args.target_data, view
                )
                _require_same_monitored_model(source_identity, target_identity)
                target = target_splits["test"]
            else:
                target = None
            train = source["train"]
            for k in k_values:
                for seed in range(args.seeds):
                    run_id = (
                        f"{args.model}__{args.source_task}__{target_task}__B1_text_tfidf"
                        f"__layer-1__{view}__k{k}__seed{seed}__balanced"
                        f"{'__selection' if args.selection_only else ''}"
                    )
                    prediction_path = _prediction_path(predictions_dir, run_id)
                    if run_id in existing and prediction_path.exists():
                        continue
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
                    selected_texts = train["texts"][selection.indices]
                    vectorizer = TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        max_features=args.max_features,
                    )
                    X_selected = vectorizer.fit_transform(selected_texts)
                    classifier = LogisticRegression(
                        C=1.0,
                        solver="liblinear",
                        max_iter=1000,
                        random_state=seed,
                    )
                    classifier.fit(X_selected, selection.y)
                    scored = {
                        "reference_calibration": _score_text_split(
                            vectorizer, classifier, reference["calibration"]
                        ),
                        "reference_holdout": _score_text_split(
                            vectorizer, classifier, reference["test"]
                        ),
                        "source_eval": _score_text_split(vectorizer, classifier, source["eval"]),
                    }
                    if not args.selection_only:
                        scored["source_test"] = _score_text_split(
                            vectorizer, classifier, source["test"]
                        )
                    if target is not None and not args.selection_only:
                        scored["target_test"] = _score_text_split(vectorizer, classifier, target)
                    n_reference_groups = require_independent_reference_groups(
                        scored["reference_calibration"]["question_ids"],
                        min_reference_groups=args.min_reference_groups,
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
                        "probe": "B1_text_tfidf",
                        "method_family": "black_box_text",
                        "k": k,
                        "k_unit": "positive_scenario_groups",
                        "seed": seed,
                        "balance_mode": "balanced",
                        "model": args.model,
                        **source_identity,
                        "layer": -1,
                        "view": view,
                        "source_task": args.source_task,
                        "target_task": target_task,
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
                        "n_train_pos": int(np.sum(selection.y == 1)),
                        "n_train_neg": int(np.sum(selection.y == 0)),
                        "n_train_groups": int(len(np.unique(selection.group_ids))),
                        "scores_are_probabilities": True,
                        "train_example_ids": train["example_ids"][selection.indices].tolist(),
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
                    for prefix in (() if "source_test" in scored else ("test",)):
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
                            row[f"{prefix}_{metric}"] = float("nan")
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

    print(f"completed {completed} text-monitor runs; saved {out_file}")


if __name__ == "__main__":
    main()
