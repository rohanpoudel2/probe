from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_cross_model(report: pd.DataFrame, metric: str, out_path: Path) -> None:
    if report.empty:
        return
    pairs = report[["source_task", "target_task"]].drop_duplicates().reset_index(drop=True)
    models = sorted(report["model"].dropna().unique())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(1, len(models))
    x = range(len(pairs))
    for idx, model in enumerate(models):
        vals = []
        lows = []
        highs = []
        for _, pair in pairs.iterrows():
            rows = report[
                (report["source_task"] == pair["source_task"])
                & (report["target_task"] == pair["target_task"])
                & (report["model"] == model)
            ]
            if rows.empty:
                vals.append(float("nan"))
                lows.append(0.0)
                highs.append(0.0)
                continue
            row = rows.iloc[0]
            val = row.get(metric, float("nan"))
            low = row.get(metric.replace("_mean", "_ci_low"), val)
            high = row.get(metric.replace("_mean", "_ci_high"), val)
            vals.append(val)
            lows.append(max(0.0, val - low))
            highs.append(max(0.0, high - val))
        offset = [(i + idx * width) for i in x]
        ax.bar(offset, vals, width=width, label=model)
        ax.errorbar(offset, vals, yerr=[lows, highs], fmt="none", capsize=3)

    centers = [i + width * (len(models) - 1) / 2 for i in x]
    labels = [f"{r.source_task}->{r.target_task}" for r in pairs.itertuples(index=False)]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title("Frozen-selector transfer comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_k_sweep(report: pd.DataFrame, metric: str, out_path: Path) -> None:
    if report.empty or "k" not in report.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped = report.groupby(["model", "k"], dropna=False)[metric].mean().reset_index()
    for model, g in grouped.groupby("model", dropna=False):
        g = g.sort_values("k")
        ax.plot(g["k"], g[metric], marker="o", label=model)
    ax.set_xlabel("k")
    ax.set_ylabel(metric)
    ax.set_title("Few-shot scaling under frozen selection")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera-ready style plots from frozen transfer report")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--metric", default="transfer_recall_at_1pct_fpr_mean")
    args = parser.parse_args()

    report_path = Path(args.results_dir) / "task_frozen_transfer_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing {report_path}. Run build_frozen_transfer_report.py first.")
    report = pd.read_csv(report_path)
    outdir = Path(args.results_dir)
    plot_cross_model(report, args.metric, outdir / "camera_ready_cross_model.png")
    plot_k_sweep(report, args.metric, outdir / "camera_ready_k_sweep.png")
    print(f"saved {outdir / 'camera_ready_cross_model.png'}")
    print(f"saved {outdir / 'camera_ready_k_sweep.png'}")


if __name__ == "__main__":
    main()
