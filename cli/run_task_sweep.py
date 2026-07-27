from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from data.task_feature_loading import infer_layers, load_feature_bundle
from evaluation.metrics import (
    alert_rate_summary,
    compute_alert_rate,
    compute_auprc,
    compute_auroc,
    compute_brier_score,
    compute_ece,
    compute_fpr_at_threshold,
    compute_recall_at_fpr,
    compute_recall_at_threshold,
    require_disjoint_reference_groups,
    require_independent_reference_groups,
    select_threshold_at_alert_rate,
)
from task_benchmark import TASK_PROBE_REGISTRY
from task_benchmark.sampling import FewShotSelection, sample_few_shot_train


def _load_existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from err
            if "run_id" in row:
                seen.add(str(row["run_id"]))
    return seen


def _score_split(probe, bundle: Dict[str, np.ndarray], view: str) -> dict[str, np.ndarray]:
    required = {view, "labels", "example_ids", "question_ids"}
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"Feature bundle is missing required arrays: {missing}")

    X = np.asarray(bundle[view])
    y = np.asarray(bundle["labels"], dtype=np.int64)
    example_ids = np.asarray(bundle["example_ids"]).astype(str)
    question_ids = np.asarray(bundle["question_ids"]).astype(str)
    if X.ndim != 2 or not (len(X) == len(y) == len(example_ids) == len(question_ids)):
        raise ValueError(
            f"Misaligned feature bundle for view={view}: X={X.shape}, labels={y.shape}, "
            f"example_ids={example_ids.shape}, question_ids={question_ids.shape}"
        )
    scores = np.asarray(probe.score(X), dtype=float)
    if scores.shape != y.shape or not np.all(np.isfinite(scores)):
        raise ValueError(f"Probe returned invalid scores with shape {scores.shape}")
    return {
        "labels": y,
        "scores": scores,
        "example_ids": example_ids,
        "question_ids": question_ids,
    }


def _metric_payload(
    prefix: str,
    scored: dict[str, np.ndarray],
    threshold: float,
    *,
    probability_scores: bool,
    max_alert_rate: float,
) -> dict[str, float]:
    y = scored["labels"]
    scores = scored["scores"]
    frozen_recall = compute_recall_at_threshold(y, scores, threshold)
    payload = {
        f"{prefix}_auroc": compute_auroc(y, scores),
        f"{prefix}_auprc": compute_auprc(y, scores),
        f"{prefix}_tpr_at_reference_alert_budget": frozen_recall,
        f"{prefix}_tpr_at_1pct_reference_alert_budget": (
            frozen_recall if np.isclose(max_alert_rate, 0.01) else float("nan")
        ),
        f"{prefix}_fpr_at_frozen_threshold": compute_fpr_at_threshold(y, scores, threshold),
        f"{prefix}_oracle_tpr_at_requested_fpr": compute_recall_at_fpr(
            y, scores, max_alert_rate
        ),
    }
    if probability_scores:
        payload[f"{prefix}_brier"] = compute_brier_score(y, scores)
        payload[f"{prefix}_ece"] = compute_ece(y, scores)
    else:
        payload[f"{prefix}_brier"] = float("nan")
        payload[f"{prefix}_ece"] = float("nan")
    return payload


def _prediction_records(
    run_id: str,
    split_name: str,
    scored: dict[str, np.ndarray],
    threshold: float,
) -> list[dict]:
    return [
        {
            "run_id": run_id,
            "split": split_name,
            "example_id": str(example_id),
            "question_id": str(question_id),
            "label": int(label),
            "score": float(score),
            "threshold": float(threshold),
            "predicted_positive": bool(score >= threshold),
        }
        for example_id, question_id, label, score in zip(
            scored["example_ids"],
            scored["question_ids"],
            scored["labels"],
            scored["scores"],
        )
    ]


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _prediction_path(predictions_dir: Path, run_id: str) -> Path:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return predictions_dir / f"{digest}.jsonl"


def _error_result(stage: str, err: Exception, elapsed: float) -> dict:
    return {
        "status": "error",
        "error": True,
        "error_stage": stage,
        "error_type": type(err).__name__,
        "error_message": str(err),
        "wall_clock_s": elapsed,
    }


def _run_one(
    probe_cls,
    source_train: Dict[str, np.ndarray],
    reference_calibration: Dict[str, np.ndarray],
    reference_holdout: Dict[str, np.ndarray],
    source_eval: Dict[str, np.ndarray],
    source_test: Optional[Dict[str, np.ndarray]],
    target_test: Optional[Dict[str, np.ndarray]],
    view: str,
    k: int,
    seed: int,
    balance_mode: str,
    max_reference_alert_rate: float,
    min_reference_groups: int,
    selection_only: bool = False,
    probe_kwargs: Optional[dict] = None,
) -> tuple[dict, dict[str, dict[str, np.ndarray]], FewShotSelection | None]:
    t0 = time.time()
    try:
        selection = sample_few_shot_train(
            source_train[view],
            source_train["labels"],
            k=k,
            seed=seed,
            balance_mode=balance_mode,
            group_ids=source_train["question_ids"],
            return_selection=True,
        )
        assert isinstance(selection, FewShotSelection)
    except Exception as err:
        return _error_result("few_shot_sampling", err, time.time() - t0), {}, None

    probe = probe_cls(**(probe_kwargs or {}))
    try:
        probe.fit(selection.X, selection.y)
    except Exception as err:
        return _error_result("probe_fit", err, time.time() - t0), {}, selection

    try:
        scored = {
            "reference_calibration": _score_split(probe, reference_calibration, view),
            "reference_holdout": _score_split(probe, reference_holdout, view),
            "source_eval": _score_split(probe, source_eval, view),
        }
        if not selection_only and source_test is not None:
            scored["source_test"] = _score_split(probe, source_test, view)
        if not selection_only and target_test is not None:
            scored["target_test"] = _score_split(probe, target_test, view)
        if np.any(scored["reference_calibration"]["labels"] != 0):
            raise ValueError(
                "Reference calibration bundles must use membership value 0 only"
            )
        if np.any(scored["reference_holdout"]["labels"] != 0):
            raise ValueError("Reference holdout bundles must use membership value 0 only")
        n_reference_groups = require_independent_reference_groups(
            scored["reference_calibration"]["question_ids"],
            min_reference_groups=min_reference_groups,
        )
        n_reference_holdout_groups = require_independent_reference_groups(
            scored["reference_holdout"]["question_ids"],
            min_reference_groups=min_reference_groups,
        )
        require_disjoint_reference_groups(
            scored["reference_calibration"]["question_ids"],
            scored["reference_holdout"]["question_ids"],
        )
        threshold = select_threshold_at_alert_rate(
            scored["reference_calibration"]["scores"],
            max_alert_rate=max_reference_alert_rate,
            min_reference=min_reference_groups,
        )
    except Exception as err:
        return _error_result("frozen_threshold", err, time.time() - t0), {}, selection

    probability_scores = bool(getattr(probe, "scores_are_probabilities", False))
    holdout_alerts = alert_rate_summary(
        scored["reference_holdout"]["scores"], threshold
    )
    row: dict = {
        "status": "ok",
        "error": False,
        "error_stage": None,
        "error_type": None,
        "error_message": None,
        "operating_threshold": float(threshold),
        "requested_max_reference_alert_rate": float(max_reference_alert_rate),
        "threshold_source": "unlabeled_reference_traffic",
        "calibration_method": "split_conformal_upper_tail_v1",
        "n_reference_calibration": int(len(scored["reference_calibration"]["scores"])),
        "n_reference_groups": n_reference_groups,
        "n_reference_holdout": int(len(scored["reference_holdout"]["scores"])),
        "n_reference_holdout_groups": n_reference_holdout_groups,
        "reference_calibration_alert_rate": compute_alert_rate(
            scored["reference_calibration"]["scores"], threshold
        ),
        "reference_holdout_alert_count": holdout_alerts["alerts"],
        "reference_holdout_alert_rate": holdout_alerts["rate"],
        "reference_holdout_alert_rate_ci_low": holdout_alerts["ci_low"],
        "reference_holdout_alert_rate_ci_high": holdout_alerts["ci_high"],
        "reference_holdout_alert_budget_violation": bool(
            holdout_alerts["ci_low"] > max_reference_alert_rate
        ),
        "conformal_p_value_resolution": 1.0
        / (len(scored["reference_calibration"]["scores"]) + 1.0),
        "n_train_pos": int(np.sum(selection.y == 1)),
        "n_train_neg": int(np.sum(selection.y == 0)),
        "n_train_groups": int(len(np.unique(selection.group_ids))),
        "scores_are_probabilities": probability_scores,
    }
    row.update(
        _metric_payload(
            "eval",
            scored["source_eval"],
            threshold,
            probability_scores=probability_scores,
            max_alert_rate=max_reference_alert_rate,
        )
    )
    if "source_test" in scored:
        row.update(
            _metric_payload(
                "test",
                scored["source_test"],
                threshold,
                probability_scores=probability_scores,
                max_alert_rate=max_reference_alert_rate,
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
                probability_scores=probability_scores,
                max_alert_rate=max_reference_alert_rate,
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
    row["wall_clock_s"] = time.time() - t0
    return row, scored, selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a group-aware task-monitoring sweep")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--source_task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--reference_split", default="calibration")
    parser.add_argument("--reference_holdout_split", default="test")
    parser.add_argument("--target_dir", default=None)
    parser.add_argument("--target_task", default=None)
    parser.add_argument("--views", default="full_text,answer")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--probes", default="P1_logistic,P2_mass_mean,P3_lda,P4_cosine")
    parser.add_argument("--k_values", default="1,2,4,8")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--balance_modes", default="balanced")
    parser.add_argument("--max_reference_alert_rate", type=float, default=0.01)
    parser.add_argument("--min_reference_groups", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--selection_only",
        action="store_true",
        help="Score source eval and reference traffic without touching test targets.",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Return success despite failed runs; prohibited for confirmatory execution.",
    )
    args = parser.parse_args()

    views = [value.strip() for value in args.views.split(",") if value.strip()]
    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    balance_modes = [value.strip() for value in args.balance_modes.split(",") if value.strip()]
    probe_names = [value.strip() for value in args.probes.split(",") if value.strip()]
    unknown_probes = sorted(set(probe_names).difference(TASK_PROBE_REGISTRY))
    if unknown_probes:
        raise ValueError(f"Unknown probes: {unknown_probes}")

    layers = infer_layers(args.source_dir) if args.layers == "all" else [
        int(value.strip()) for value in args.layers.split(",") if value.strip()
    ]
    if not layers:
        raise ValueError(f"No activation layers found in {args.source_dir}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = results_dir / "predictions"
    stage_suffix = "__selection" if args.selection_only else ""
    out_file = (
        results_dir
        / f"{args.source_task}__to__{args.target_task or args.source_task}{stage_suffix}.jsonl"
    )
    existing_run_ids = set() if args.overwrite else _load_existing_run_ids(out_file)
    if args.overwrite and out_file.exists():
        out_file.unlink()

    total = completed = failed = skipped_unsupported = 0
    bundle_cache: Dict[tuple[str, str, int], Dict[str, np.ndarray]] = {}

    def _load_cached(features_dir: str, split: str, layer: int) -> Dict[str, np.ndarray]:
        key = (features_dir, split, layer)
        if key not in bundle_cache:
            bundle_cache[key] = load_feature_bundle(features_dir, split, layer)
        return bundle_cache[key]

    reference_dir = args.reference_dir
    summary_rows_written = 0
    summary_fsync_every = 128
    with out_file.open("a", encoding="utf-8") as summary_handle:
        for layer in layers:
            # Feature bundles are keyed by (directory, split, layer) and never
            # reused across layers, so release the previous layer's arrays before
            # loading this one. This bounds host/unified memory to a single
            # layer's bundles instead of accumulating every layer's.
            bundle_cache.clear()
            for probe_name in probe_names:
                probe_cls = TASK_PROBE_REGISTRY[probe_name]
                source_train = _load_cached(args.source_dir, "train", layer)
                reference_calibration = _load_cached(
                    reference_dir, args.reference_split, layer
                )
                reference_holdout = _load_cached(
                    reference_dir, args.reference_holdout_split, layer
                )
                source_eval = _load_cached(args.source_dir, "eval", layer)
                source_test = (
                    None
                    if args.selection_only
                    else _load_cached(args.source_dir, "test", layer)
                )
                target_test = (
                    _load_cached(args.target_dir, "test", layer)
                    if args.target_dir and not args.selection_only
                    else None
                )

                missing_views = [
                    view
                    for view in views
                    if any(
                        view not in bundle
                        for bundle in (
                            source_train,
                            reference_calibration,
                            reference_holdout,
                            source_eval,
                            *(() if source_test is None else (source_test,)),
                            *(() if target_test is None else (target_test,)),
                        )
                    )
                ]
                if missing_views:
                    raise ValueError(
                        f"Layer {layer} probe {probe_name} is missing requested views {missing_views}"
                    )

                probe_kwargs = None
                for view in views:
                    for k in k_values:
                        for balance_mode in balance_modes:
                            minimum_counts = getattr(
                                probe_cls, "minimum_class_counts", {0: 1, 1: 1}
                            )
                            if balance_mode == "balanced" and any(
                                k < required for required in minimum_counts.values()
                            ):
                                skipped_unsupported += args.seeds
                                continue
                            for seed in range(args.seeds):
                                total += 1
                                run_id = (
                                    f"{args.model}__{args.source_task}__{args.target_task or args.source_task}"
                                    f"__{probe_name}__layer{layer}__{view}__k{k}__seed{seed}__{balance_mode}"
                                    f"{'__selection' if args.selection_only else ''}"
                                )
                                prediction_path = _prediction_path(predictions_dir, run_id)
                                if run_id in existing_run_ids and prediction_path.exists():
                                    continue
                                row, scored, selection = _run_one(
                                    probe_cls=probe_cls,
                                    source_train=source_train,
                                    reference_calibration=reference_calibration,
                                    reference_holdout=reference_holdout,
                                    source_eval=source_eval,
                                    source_test=source_test,
                                    target_test=target_test,
                                    view=view,
                                    k=k,
                                    seed=seed,
                                    balance_mode=balance_mode,
                                    max_reference_alert_rate=args.max_reference_alert_rate,
                                    min_reference_groups=args.min_reference_groups,
                                    selection_only=args.selection_only,
                                    probe_kwargs=probe_kwargs,
                                )
                                row.update(
                                    {
                                        "run_id": run_id,
                                        "execution_mode": (
                                            "selection"
                                            if args.selection_only
                                            else "confirmatory"
                                        ),
                                        "probe": probe_name,
                                        "k": k,
                                        "k_unit": "positive_scenario_groups",
                                        "seed": seed,
                                        "balance_mode": balance_mode,
                                        "model": args.model,
                                        "layer": layer,
                                        "view": view,
                                        "source_task": args.source_task,
                                        "target_task": args.target_task or args.source_task,
                                    }
                                )
                                if row["status"] == "ok":
                                    prediction_rows: list[dict] = []
                                    for split_name, split_scores in scored.items():
                                        prediction_rows.extend(
                                            _prediction_records(
                                                run_id,
                                                split_name,
                                                split_scores,
                                                row["operating_threshold"],
                                            )
                                        )
                                    if selection is not None:
                                        row["train_example_ids"] = source_train["example_ids"][
                                            selection.indices
                                        ].astype(str).tolist()
                                        row["train_question_ids"] = selection.group_ids.astype(str).tolist()
                                    _atomic_write_jsonl(prediction_path, prediction_rows)
                                    row["prediction_file"] = str(prediction_path)
                                    completed += 1
                                else:
                                    row["prediction_file"] = None
                                    failed += 1

                                summary_handle.write(json.dumps(row, sort_keys=True) + "\n")
                                # Flush every row so a resumed process sees every
                                # completed run, but only force the summary to
                                # stable storage periodically: prediction files
                                # are written atomically-and-fsynced before their
                                # summary row, and any summary tail lost to a hard
                                # crash is recomputed on resume.
                                summary_handle.flush()
                                summary_rows_written += 1
                                if summary_rows_written % summary_fsync_every == 0:
                                    os.fsync(summary_handle.fileno())
                                existing_run_ids.add(run_id)
        # Durably persist the full summary once every run has been recorded.
        summary_handle.flush()
        os.fsync(summary_handle.fileno())

    print(
        f"completed {completed} valid runs; failed {failed}; "
        f"skipped unsupported {skipped_unsupported}; considered {total}"
    )
    print(f"saved results to {out_file}")
    if failed and not args.allow_partial:
        raise RuntimeError(
            f"{failed} runs failed. Inspect error_stage/error_message; confirmatory runs forbid partial success."
        )


if __name__ == "__main__":
    main()
