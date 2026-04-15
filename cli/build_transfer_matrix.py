from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Build transfer matrix table and heatmap from best task systems")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--metric", default="transfer_recall_at_1pct_fpr_mean")
    args = parser.parse_args()

    best_path = Path(args.results_dir) / "task_best_view_layer.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing {best_path}. Run aggregate_task_results.py first.")

    best = pd.read_csv(best_path)
    if best.empty:
        print("No best-system rows found.")
        return

    pivot = best.pivot_table(
        index="source_task",
        columns="target_task",
        values=args.metric,
        aggfunc="max",
    )
    csv_path = Path(args.results_dir) / "task_transfer_matrix.csv"
    pivot.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    arr = pivot.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(args.metric)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            text = "nan" if pd.isna(val) else f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    png_path = Path(args.results_dir) / "task_transfer_matrix.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {csv_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
