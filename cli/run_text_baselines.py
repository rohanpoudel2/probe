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
    require_independent_calibration_negatives,
    select_threshold_at_fpr,
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
    # Skip splits with no examples (e.g. a task with a dedicated benign calibration
    # source has an empty task-internal calibration split); building a text view
    # from an empty split would fail and such splits are never consumed here.
    return (
        {
            name: examples_to_text_arrays(rows, view)
            for name, rows in splits.items()
            if rows
        },
        monitored_model_identity(examples),
    )


def _load_calibration(task_name: str, path: str, view: str) -> dict[str, np.ndarray]:
    task = TASK_REGISTRY[task_name]()
    examples = task.load(path)
    if any(example.metadata.get("data_origin") != "on_policy_generation" for example in examples):
        raise ValueError("Calibration data must contain only on-policy model outputs")
    if any(example.label != 0 for example in examples):
        raise ValueError("Dedicated benign calibration data must be all-negative")
    return examples_to_text_arrays(examples, view), monitored_model_identity(examples)


def _require_same_monitored_model(reference: dict[str, str], other: dict[str, str]) -> None:
    if reference != other:
        raise ValueError(
            "Source, calibration, and transfer text data must come from the same "
            "monitored model and tokenizer revisions"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched TF-IDF black-box monitor baselines")
    parser.add_argument("--source_task", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--source_data", required=True)
    parser.add_argument("--target_task", default=None, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--target_data", default=None)
    parser.add_argument("--calibration_task", default=None, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--calibration_data", default=None)
    parser.add_argument("--model", required=True, help="Model that generated the monitored rollouts")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--views", default="prompt_text,answer_text,transcript_text")
    parser.add_argument("--k_values", default="1,2,4,8,16,32")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_fpr", type=float, default=0.01)
    parser.add_argument("--min_calibration_negatives", type=int, default=1000)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if bool(args.target_task) != bool(args.target_data):
        raise ValueError("--target_task and --target_data must be provided together")
    if bool(args.calibration_task) != bool(args.calibration_data):
        raise ValueError("--calibration_task and --calibration_data must be provided together")
    if args.source_task == "benign_calibration" or args.target_task == "benign_calibration":
        raise ValueError("benign_calibration can only be supplied through --calibration_task")
    if args.calibration_task and args.calibration_task != "benign_calibration":
        raise ValueError("Dedicated calibration must use the benign_calibration task contract")
    views = [value.strip() for value in args.views.split(",") if value.strip()]
    if not set(views).issubset(ALLOWED_TEXT_VIEWS):
        raise ValueError(f"Text views must be chosen from {sorted(ALLOWED_TEXT_VIEWS)}")
    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    target_task = args.target_task or args.source_task

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{args.source_task}__to__{target_task}__text_baselines.jsonl"
    if args.overwrite and out_file.exists():
        out_file.unlink()
    existing = _load_existing_run_ids(out_file)
    predictions_dir = results_dir / "predictions"
    completed = 0

    with out_file.open("a", encoding="utf-8") as handle:
        for view in views:
            source, source_identity = _load_splits(args.source_task, args.source_data, view)
            if args.calibration_data:
                calibration, calibration_identity = _load_calibration(
                    args.calibration_task, args.calibration_data, view
                )
                _require_same_monitored_model(source_identity, calibration_identity)
            else:
                calibration = source["calibration"]
            if args.target_data:
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
                        "source_calibration": _score_text_split(vectorizer, classifier, calibration),
                        "source_eval": _score_text_split(vectorizer, classifier, source["eval"]),
                        "source_test": _score_text_split(vectorizer, classifier, source["test"]),
                    }
                    if target is not None:
                        scored["target_test"] = _score_text_split(vectorizer, classifier, target)
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
                    row = {
                        "status": "ok",
                        "error": False,
                        "run_id": run_id,
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

    print(f"completed {completed} text-monitor runs; saved {out_file}")


if __name__ == "__main__":
    main()
