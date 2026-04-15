from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation.task_selection import apply_frozen_selection, select_frozen_source_systems


def main() -> None:
    parser = argparse.ArgumentParser(description="Build no-leakage frozen transfer report from task_summary.csv")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--selection_metric", default="eval_recall_at_1pct_fpr_mean")
    args = parser.parse_args()

    summary_path = Path(args.results_dir) / "task_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run aggregate_task_results.py first.")

    summary = pd.read_csv(summary_path)
    selected = select_frozen_source_systems(summary, selection_metric=args.selection_metric)
    report = apply_frozen_selection(summary, selected)

    outdir = Path(args.results_dir)
    selected.to_csv(outdir / "task_frozen_source_systems.csv", index=False)
    report.to_csv(outdir / "task_frozen_transfer_report.csv", index=False)

    print(f"saved {outdir / 'task_frozen_source_systems.csv'}")
    print(f"saved {outdir / 'task_frozen_transfer_report.csv'}")


if __name__ == "__main__":
    main()
