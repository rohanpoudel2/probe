from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-model and cross-seed robustness summaries")
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary = pd.read_csv(results_dir / "task_summary.csv")

    group_cols = ["model", "source_task", "target_task"]
    metrics = [
        "eval_recall_at_1pct_fpr_mean",
        "test_recall_at_1pct_fpr_mean",
        "transfer_recall_at_1pct_fpr_mean",
        "transfer_auroc_mean",
    ]
    rows = []
    for keys, g in summary.groupby(group_cols, dropna=False):
        row = {col: val for col, val in zip(group_cols, keys)}
        row["num_systems"] = int(len(g))
        for metric in metrics:
            if metric in g.columns:
                row[f"{metric}_mean_across_systems"] = float(g[metric].mean())
                row[f"{metric}_std_across_systems"] = float(g[metric].std(ddof=0))
        rows.append(row)

    out = pd.DataFrame(rows)
    steering_path = results_dir / "task_steering_best.csv"
    if steering_path.exists() and steering_path.stat().st_size > 0:
        steering = pd.read_csv(steering_path)
        if not steering.empty:
            steering_grouped = steering.groupby(group_cols, dropna=False)["selectivity_score"].mean().reset_index()
            out = out.merge(steering_grouped, how="left", on=group_cols)
    out_path = results_dir / "robustness_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
