"""Aggregate result rows into paper ready summary tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import compute_fsei


def collect_results(results_dir: str) -> pd.DataFrame:
    results_path = Path(results_dir)
    rows = []
    for f in sorted(results_path.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across seeds for each experiment configuration."""
    if df.empty:
        return pd.DataFrame()

    group_cols = ["probe", "k", "balance_mode", "model", "layer", "dataset"]
    metric_cols = [
        "eval_auroc",
        "eval_recall_at_1pct_fpr",
        "test_auroc",
        "test_recall_at_1pct_fpr",
        "ood_auroc",
        "ood_recall_at_1pct_fpr",
        "wall_clock_s",
    ]

    agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in agg.columns
    ]
    return agg


def select_best_layer(
    summary: pd.DataFrame,
    selection_metric: str = "eval_recall_at_1pct_fpr_mean",
) -> pd.DataFrame:
    """Pick best layer separately for each probe, k, balance mode, model, dataset."""
    if summary.empty:
        return pd.DataFrame()

    idx = summary.groupby(
        ["probe", "k", "balance_mode", "model", "dataset"],
        dropna=False,
    )[selection_metric].idxmax()

    return summary.loc[idx].reset_index(drop=True)


def compute_fsei_table(best_layer_summary: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """Compute FSEI using best layer test recall values."""
    rows = []

    if best_layer_summary.empty:
        return pd.DataFrame()

    for keys, g in best_layer_summary.groupby(
        ["probe", "balance_mode", "model", "dataset"],
        dropna=False,
    ):
        recall_by_k = {}
        best_layer_by_k = {}

        for _, row in g.iterrows():
            k = int(row["k"])
            val = row["test_recall_at_1pct_fpr_mean"]
            if pd.notnull(val):
                recall_by_k[k] = float(val)
                best_layer_by_k[k] = int(row["layer"])

        available_k = [k for k in k_values if k in recall_by_k]
        fsei = compute_fsei(recall_by_k, available_k) if len(available_k) >= 2 else float("nan")

        rows.append(
            {
                "probe": keys[0],
                "balance_mode": keys[1],
                "model": keys[2],
                "dataset": keys[3],
                "fsei": fsei,
                "k_min": min(available_k) if available_k else np.nan,
                "k_max": max(available_k) if available_k else np.nan,
            }
        )

    return pd.DataFrame(rows)


def make_decision_table(best_layer_summary: pd.DataFrame) -> pd.DataFrame:
    """For each dataset, model, balance mode, and k, recommend the best probe."""
    if best_layer_summary.empty:
        return pd.DataFrame()

    idx = best_layer_summary.groupby(
        ["dataset", "model", "balance_mode", "k"],
        dropna=False,
    )["test_recall_at_1pct_fpr_mean"].idxmax()

    cols = [
        "dataset",
        "model",
        "balance_mode",
        "k",
        "probe",
        "layer",
        "test_recall_at_1pct_fpr_mean",
        "test_recall_at_1pct_fpr_std",
        "test_auroc_mean",
        "test_auroc_std",
        "ood_recall_at_1pct_fpr_mean",
        "ood_recall_at_1pct_fpr_std",
        "ood_auroc_mean",
        "ood_auroc_std",
    ]

    keep_cols = [c for c in cols if c in best_layer_summary.columns]
    return (
        best_layer_summary.loc[idx, keep_cols]
        .sort_values(["dataset", "model", "balance_mode", "k"])
        .reset_index(drop=True)
    )


def make_ood_table(best_layer_summary: pd.DataFrame) -> pd.DataFrame:
    if best_layer_summary.empty:
        return pd.DataFrame()

    cols = [
        "probe",
        "k",
        "balance_mode",
        "model",
        "dataset",
        "layer",
        "ood_auroc_mean",
        "ood_auroc_std",
        "ood_recall_at_1pct_fpr_mean",
        "ood_recall_at_1pct_fpr_std",
    ]
    keep_cols = [c for c in cols if c in best_layer_summary.columns]
    return best_layer_summary[keep_cols].copy()


def make_layer_choices(best_layer_summary: pd.DataFrame) -> pd.DataFrame:
    if best_layer_summary.empty:
        return pd.DataFrame()

    cols = [
        "probe",
        "k",
        "balance_mode",
        "model",
        "dataset",
        "layer",
        "eval_recall_at_1pct_fpr_mean",
        "test_recall_at_1pct_fpr_mean",
    ]
    keep_cols = [c for c in cols if c in best_layer_summary.columns]
    return best_layer_summary[keep_cols].sort_values(
        ["dataset", "model", "probe", "balance_mode", "k"]
    ).reset_index(drop=True)
