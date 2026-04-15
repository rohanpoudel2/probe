from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_best_systems(best: pd.DataFrame, metric: str, out_path: Path) -> None:
    if best.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    df = best.copy()
    df["label"] = df["source_task"] + "->" + df["target_task"] + "\n" + df["probe"] + ", L" + df["layer"].astype(str) + ", " + df["view"]
    values = df[metric].fillna(df.get("test_recall_at_1pct_fpr_mean", 0.0))
    yerr_low = values - df.get(metric.replace("_mean", "_ci_low"), values)
    yerr_high = df.get(metric.replace("_mean", "_ci_high"), values) - values
    ax.bar(range(len(df)), values)
    ax.errorbar(range(len(df)), values, yerr=[yerr_low, yerr_high], fmt="none", capsize=4)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=35, ha="right")
    ax.set_ylabel(metric)
    ax.set_title("Best systems by task pair")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-ready task benchmark summaries")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--metric", default="transfer_recall_at_1pct_fpr_mean")
    args = parser.parse_args()

    best_path = Path(args.results_dir) / "task_best_view_layer.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing {best_path}. Run aggregate_task_results.py first.")
    best = pd.read_csv(best_path)
    out_path = Path(args.results_dir) / "best_systems_barplot.png"
    plot_best_systems(best, args.metric, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
