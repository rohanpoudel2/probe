from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.aggregation import collect_results
from evaluation.task_aggregation import (
    compute_task_fsei,
    make_transfer_table,
    make_view_layer_table,
    select_best_view_layer,
)
from evaluation.task_model_selection import rank_models
from evaluation.task_statistics import add_seed_summary_columns
from data.outcomes import MODEL_OUTCOME_CLASSES


GROUP_COLS = [
    "probe",
    "k",
    "balance_mode",
    "source_task",
    "target_task",
    "model",
    "layer",
    "view",
]

METRIC_COLS = [
    "eval_auroc",
    "eval_auprc",
    "eval_tpr_at_1pct_reference_alert_budget",
    "eval_tpr_at_reference_alert_budget",
    "eval_fpr_at_frozen_threshold",
    "eval_brier",
    "eval_ece",
    "test_auroc",
    "test_auprc",
    "test_tpr_at_1pct_reference_alert_budget",
    "test_tpr_at_reference_alert_budget",
    "test_fpr_at_frozen_threshold",
    "test_brier",
    "test_ece",
    "transfer_auroc",
    "transfer_auprc",
    "transfer_tpr_at_1pct_reference_alert_budget",
    "transfer_tpr_at_reference_alert_budget",
    "transfer_fpr_at_frozen_threshold",
    "transfer_brier",
    "transfer_ece",
    "reference_calibration_alert_rate",
    "reference_holdout_alert_rate",
    "wall_clock_s",
]

_OUTCOME_CLASSES = MODEL_OUTCOME_CLASSES
_OUTCOME_METRIC_BASES = (
    "tpr_at_reference_alert_budget",
    "tpr_at_1pct_reference_alert_budget",
    "fpr_at_frozen_threshold",
    "oracle_tpr_at_requested_fpr",
    "auroc",
    "auprc",
    "brier",
    "ece",
    "n_examples",
    "n_positive",
    "n_negative",
)
for _prefix in ("eval", "test", "transfer"):
    for _outcome in _OUTCOME_CLASSES:
        for _base in _OUTCOME_METRIC_BASES:
            METRIC_COLS.append(f"{_prefix}_{_outcome}_{_base}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate structured task-benchmark results"
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument(
        "--selection_metric",
        default="eval_tpr_at_1pct_reference_alert_budget_mean",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    args = parser.parse_args()

    df = collect_results(args.results_dir)
    if df.empty:
        print("No results found.")
        return

    if "status" in df.columns:
        failed = df[df["status"] != "ok"]
        if not failed.empty:
            raise RuntimeError(
                f"Refusing to aggregate {len(failed)} failed/partial runs; repair or explicitly remove them"
            )
    summary = add_seed_summary_columns(
        df,
        group_cols=GROUP_COLS,
        metric_cols=METRIC_COLS,
    )
    best = select_best_view_layer(summary, selection_metric=args.selection_metric)
    view_layer = make_view_layer_table(best)
    transfer = make_transfer_table(best)
    task_fsei = compute_task_fsei(best)
    ranking = rank_models(best)

    outdir = Path(args.results_dir)
    summary.to_csv(outdir / "task_summary.csv", index=False)
    best.to_csv(outdir / "task_best_view_layer.csv", index=False)
    view_layer.to_csv(outdir / "task_view_layer_table.csv", index=False)
    transfer.to_csv(outdir / "task_transfer_table.csv", index=False)
    task_fsei.to_csv(outdir / "task_fsei.csv", index=False)
    ranking.to_csv(outdir / "task_model_ranking.csv", index=False)

    print(f"saved {outdir / 'task_summary.csv'}")
    print(f"saved {outdir / 'task_best_view_layer.csv'}")
    print(f"saved {outdir / 'task_view_layer_table.csv'}")
    print(f"saved {outdir / 'task_transfer_table.csv'}")
    print(f"saved {outdir / 'task_fsei.csv'}")
    print(f"saved {outdir / 'task_model_ranking.csv'}")


if __name__ == "__main__":
    main()
