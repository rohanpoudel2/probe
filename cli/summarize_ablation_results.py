from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ablation result directories")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument(
        "--metric", default="transfer_tpr_at_1pct_reference_alert_budget_mean"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path; defaults to <root_dir>/ablation_summary.csv.",
    )
    args = parser.parse_args()

    rows = []
    root = Path(args.root_dir)
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        report_path = subdir / "task_cross_task_transfer.csv"
        if not report_path.exists():
            continue
        df = pd.read_csv(report_path)
        if df.empty:
            continue
        rows.append({
            "ablation": subdir.name,
            "mean_metric": float(df[args.metric].mean()),
            "max_metric": float(df[args.metric].max()),
            "num_rows": int(len(df)),
        })
    out = pd.DataFrame(rows).sort_values("mean_metric", ascending=False)
    out_path = Path(args.output) if args.output else root / "ablation_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
