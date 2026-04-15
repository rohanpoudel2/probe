from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged cross-model tables from frozen transfer report")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--metric", default="transfer_recall_at_1pct_fpr_mean")
    args = parser.parse_args()

    report_path = Path(args.results_dir) / "task_frozen_transfer_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing {report_path}. Run build_frozen_transfer_report.py first.")

    report = pd.read_csv(report_path)
    metric = args.metric
    pivot = report.pivot_table(index=["source_task", "target_task"], columns="model", values=metric, aggfunc="max")
    rank = report[["model", "source_task", "target_task", metric]].copy()
    rank = rank.sort_values(["source_task", "target_task", metric], ascending=[True, True, False]).reset_index(drop=True)
    rank["rank_within_pair"] = rank.groupby(["source_task", "target_task"]).cumcount() + 1

    outdir = Path(args.results_dir)
    pivot.to_csv(outdir / "task_cross_model_table.csv")
    rank.to_csv(outdir / "task_cross_model_rankings.csv", index=False)

    print(f"saved {outdir / 'task_cross_model_table.csv'}")
    print(f"saved {outdir / 'task_cross_model_rankings.csv'}")


if __name__ == "__main__":
    main()
