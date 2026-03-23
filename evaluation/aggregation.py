"""Results aggregation: collect JSON lines into summary DataFrames."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import compute_fsei


def collect_results(results_dir: str) -> pd.DataFrame:
    """Load all result JSON lines files into a single DataFrame."""
    results_path = Path(results_dir)
    rows = []
    for f in sorted(results_path.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std across seeds for each experiment configuration."""
    group_cols = ["probe", "k", "balance_mode", "model", "layer", "dataset"]
    metric_cols = ["auroc", "recall_at_1pct_fpr"]

    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    # Flatten multi-level columns
    agg.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in agg.columns
    ]
    return agg


def select_best_layer(summary: pd.DataFrame) -> pd.DataFrame:
    """For each (probe, balance_mode, model, dataset), pick the layer with
    the highest mean AUROC on the eval set. This avoids layer selection
    as a confound in the final comparison."""
    idx = summary.groupby(
        ["probe", "balance_mode", "model", "dataset"]
    )["auroc_mean"].idxmax()
    return summary.loc[idx].reset_index(drop=True)


def compute_fsei_table(
    df: pd.DataFrame, k_values: list[int]
) -> pd.DataFrame:
    """Compute FSEI for each (probe, balance_mode, model, dataset) at the best layer."""
    # First get mean recall@1%FPR per (probe, k, balance_mode, model, layer, dataset)
    group_cols = ["probe", "k", "balance_mode", "model", "layer", "dataset"]
    means = df.groupby(group_cols)["recall_at_1pct_fpr"].mean().reset_index()

    # Select best layer per probe config
    best_layers = means.groupby(
        ["probe", "balance_mode", "model", "dataset"]
    ).apply(
        lambda g: g.loc[
            g.groupby("layer")["recall_at_1pct_fpr"].mean().idxmax()
        ]["layer"].iloc[0]
        if len(g) > 0 else None,
        include_groups=False,
    ).reset_index(name="best_layer")

    # Compute FSEI per probe config
    fsei_rows = []
    for _, row in best_layers.iterrows():
        probe, bm, model, ds, layer = (
            row["probe"], row["balance_mode"], row["model"],
            row["dataset"], row["best_layer"],
        )
        subset = means[
            (means["probe"] == probe)
            & (means["balance_mode"] == bm)
            & (means["model"] == model)
            & (means["dataset"] == ds)
            & (means["layer"] == layer)
        ]
        recall_by_k = dict(zip(subset["k"], subset["recall_at_1pct_fpr"]))
        available_k = [k for k in k_values if k in recall_by_k]

        if len(available_k) >= 2:
            fsei = compute_fsei(recall_by_k, available_k)
        else:
            fsei = float("nan")

        fsei_rows.append({
            "probe": probe,
            "balance_mode": bm,
            "model": model,
            "dataset": ds,
            "best_layer": layer,
            "fsei": fsei,
        })

    return pd.DataFrame(fsei_rows)


def make_decision_table(summary: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """For each k value, which probe achieves the best mean Recall@1%FPR?

    This is the practitioner deliverable: a lookup table from k to recommended probe.
    """
    rows = []
    for k in k_values:
        subset = summary[summary["k"] == k]
        if subset.empty:
            continue
        best_idx = subset["recall_at_1pct_fpr_mean"].idxmax()
        best = subset.loc[best_idx]
        rows.append({
            "k": k,
            "best_probe": best["probe"],
            "recall_at_1pct_fpr": best["recall_at_1pct_fpr_mean"],
            "auroc": best["auroc_mean"],
            "balance_mode": best["balance_mode"],
        })
    return pd.DataFrame(rows)
