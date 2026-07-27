from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cross-model tables for source-selected primary systems"
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument(
        "--metric", default="transfer_tpr_at_1pct_reference_alert_budget_mean"
    )
    args = parser.parse_args()

    report_path = Path(args.results_dir) / "task_primary_transfer_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing {report_path}. Run build_frozen_transfer_report.py first."
        )

    report = pd.read_csv(report_path)
    metric = args.metric
    required = {
        "model",
        "source_task",
        "target_task",
        "access_regime",
        "probe",
        "k",
        metric,
    }
    missing = sorted(required.difference(report.columns))
    if missing:
        raise ValueError(f"Primary transfer report lacks columns {missing}")

    identity = [
        "model",
        "source_task",
        "target_task",
        "access_regime",
        "k",
    ]
    if report.duplicated(identity).any():
        duplicates = report.loc[report.duplicated(identity, keep=False), identity]
        raise ValueError(
            "Primary transfer report contains multiple source-selected systems for "
            f"one model/task/access/k identity: "
            f"{duplicates.head(5).to_dict(orient='records')}"
        )

    pivot = report.pivot(
        index=["source_task", "target_task", "access_regime", "k"],
        columns="model",
        values=metric,
    )
    rank = report[
        [
            "model",
            "source_task",
            "target_task",
            "access_regime",
            "probe",
            "k",
            metric,
        ]
    ].copy()
    rank = rank.sort_values(
        ["source_task", "target_task", "access_regime", "k", metric],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)
    rank["rank_within_pair"] = (
        rank.groupby(
            ["source_task", "target_task", "access_regime", "k"]
        ).cumcount()
        + 1
    )

    outdir = Path(args.results_dir)
    pivot.to_csv(outdir / "task_cross_model_table.csv")
    rank.to_csv(outdir / "task_cross_model_rankings.csv", index=False)

    print(f"saved {outdir / 'task_cross_model_table.csv'}")
    print(f"saved {outdir / 'task_cross_model_rankings.csv'}")


if __name__ == "__main__":
    main()
