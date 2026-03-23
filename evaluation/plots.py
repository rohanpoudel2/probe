"""Generate paper-ready plots from analyzed benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_k(best: pd.DataFrame, metric: str, output_path: Path, title: str):
    if best.empty or metric not in best.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=best,
        x="k",
        y=metric,
        hue="probe",
        style="dataset",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " "))
    _save(fig, output_path)


def plot_fsei(fsei: pd.DataFrame, output_path: Path):
    if fsei.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ranked = fsei.sort_values("fsei", ascending=False)
    sns.barplot(data=ranked, x="fsei", y="probe", hue="dataset", ax=ax)
    ax.set_title("Few-Shot Efficiency Index by Probe")
    _save(fig, output_path)


def plot_win_counts(decision: pd.DataFrame, output_path: Path):
    if decision.empty:
        return
    counts = (
        decision.groupby(["dataset", "model", "probe"])
        .size()
        .reset_index(name="wins")
    )
    pivot = counts.pivot_table(index="probe", columns=["dataset", "model"], values="wins", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="crest", ax=ax)
    ax.set_title("Decision-Table Win Counts")
    _save(fig, output_path)


def plot_selected_layer(layer_choices: pd.DataFrame, output_path: Path):
    if layer_choices.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=layer_choices,
        x="k",
        y="layer",
        hue="probe",
        style="dataset",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title("Selected Layer versus k")
    _save(fig, output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate plots from benchmark CSV outputs")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"

    best = pd.read_csv(results_dir / "best_layer_summary.csv") if (results_dir / "best_layer_summary.csv").exists() else pd.DataFrame()
    decision = pd.read_csv(results_dir / "decision_table.csv") if (results_dir / "decision_table.csv").exists() else pd.DataFrame()
    fsei = pd.read_csv(results_dir / "fsei.csv") if (results_dir / "fsei.csv").exists() else pd.DataFrame()
    layer_choices = pd.read_csv(results_dir / "layer_choices.csv") if (results_dir / "layer_choices.csv").exists() else pd.DataFrame()

    plot_metric_vs_k(
        best,
        "test_recall_at_1pct_fpr_mean",
        figures_dir / "recall_vs_k.png",
        "Recall@1%FPR versus k",
    )
    plot_metric_vs_k(
        best,
        "test_auroc_mean",
        figures_dir / "auroc_vs_k.png",
        "AUROC versus k",
    )
    plot_metric_vs_k(
        best,
        "ood_recall_at_1pct_fpr_mean",
        figures_dir / "ood_recall_vs_k.png",
        "OOD Recall@1%FPR versus k",
    )
    plot_fsei(fsei, figures_dir / "fsei_bar.png")
    plot_win_counts(decision, figures_dir / "win_count_heatmap.png")
    plot_selected_layer(layer_choices, figures_dir / "selected_layer_vs_k.png")


if __name__ == "__main__":
    main()
