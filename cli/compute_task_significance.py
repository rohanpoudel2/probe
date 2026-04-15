from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation.aggregation import collect_results
from evaluation.task_statistics import paired_permutation_pvalue


def _make_system_name(df: pd.DataFrame) -> pd.Series:
    return (
        df["probe"].astype(str)
        + "|L" + df["layer"].astype(str)
        + "|" + df["view"].astype(str)
        + "|k" + df["k"].astype(str)
        + "|" + df["balance_mode"].astype(str)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired significance between top two systems per task pair")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--metric", default="transfer_recall_at_1pct_fpr")
    args = parser.parse_args()

    df = collect_results(args.results_dir)
    if df.empty:
        print("No results found.")
        return

    df = df.copy()
    df["system_name"] = _make_system_name(df)

    out_rows = []
    group_cols = ["model", "source_task", "target_task"]
    for group_key, g in df.groupby(group_cols, dropna=False):
        means = g.groupby("system_name", dropna=False)[args.metric].mean().sort_values(ascending=False)
        if len(means) < 2:
            continue
        top_system, runner_up = means.index[0], means.index[1]
        top_vals = g[g["system_name"] == top_system].sort_values("seed")[args.metric].tolist()
        runner_vals = g[g["system_name"] == runner_up].sort_values("seed")[args.metric].tolist()
        p = paired_permutation_pvalue(top_vals, runner_vals)
        out_rows.append(
            {
                "model": group_key[0],
                "source_task": group_key[1],
                "target_task": group_key[2],
                "metric": args.metric,
                "top_system": top_system,
                "runner_up_system": runner_up,
                "top_mean": float(means.iloc[0]),
                "runner_up_mean": float(means.iloc[1]),
                "mean_gap": float(means.iloc[0] - means.iloc[1]),
                "paired_permutation_pvalue": p,
            }
        )

    out = pd.DataFrame(out_rows)
    out_path = Path(args.results_dir) / "task_significance.csv"
    out.to_csv(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
