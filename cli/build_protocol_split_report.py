from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split the source-selected primary-system report into same-task and "
            "cross-task protocol tables"
        )
    )
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    report_path = Path(args.results_dir) / "task_primary_transfer_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing {report_path}. Run build_frozen_transfer_report.py first."
        )

    report = pd.read_csv(report_path)
    calibration = report[report["source_task"] == report["target_task"]].copy()
    transfer = report[report["source_task"] != report["target_task"]].copy()

    outdir = Path(args.results_dir)
    calibration.to_csv(outdir / "task_same_task_calibration.csv", index=False)
    transfer.to_csv(outdir / "task_cross_task_transfer.csv", index=False)

    print(f"saved {outdir / 'task_same_task_calibration.csv'}")
    print(f"saved {outdir / 'task_cross_task_transfer.csv'}")


if __name__ == "__main__":
    main()
