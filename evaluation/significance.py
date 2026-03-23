"""Bootstrap significance testing for probe comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import compute_auroc, compute_recall_at_fpr


def paired_bootstrap_metric_diff(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_name: str = "recall_at_1pct_fpr",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)

    if metric_name == "recall_at_1pct_fpr":
        metric_fn = lambda y, s: compute_recall_at_fpr(y, s, max_fpr=0.01)
    elif metric_name == "auroc":
        metric_fn = compute_auroc
    else:
        raise ValueError(f"Unknown metric_name: {metric_name}")

    base_a = metric_fn(y_true, scores_a)
    base_b = metric_fn(y_true, scores_b)

    diffs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        sa = scores_a[idx]
        sb = scores_b[idx]
        diffs.append(metric_fn(yb, sa) - metric_fn(yb, sb))

    diffs = np.asarray(diffs, dtype=float)
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())

    return {
        "metric": metric_name,
        "score_a": float(base_a),
        "score_b": float(base_b),
        "mean_diff": float(np.nanmean(diffs)),
        "ci_low": float(np.nanpercentile(diffs, 2.5)),
        "ci_high": float(np.nanpercentile(diffs, 97.5)),
        "p_value": float(p_value),
    }


def load_prediction_artifact(predictions_dir: str | Path, run_id: str) -> dict:
    path = Path(predictions_dir) / f"{run_id}.npz"
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


def compare_two_runs(
    predictions_dir: str | Path,
    run_id_a: str,
    run_id_b: str,
    split: str = "test",
    metric_name: str = "recall_at_1pct_fpr",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    a = load_prediction_artifact(predictions_dir, run_id_a)
    b = load_prediction_artifact(predictions_dir, run_id_b)

    y_key = f"y_{split}"
    s_key = f"{split}_scores"

    if y_key not in a or y_key not in b or s_key not in a or s_key not in b:
        raise ValueError(f"Missing {split} predictions for one of the runs")

    if not np.array_equal(a[y_key], b[y_key]):
        raise ValueError("Prediction artifacts do not share aligned labels")

    out = paired_bootstrap_metric_diff(
        y_true=a[y_key],
        scores_a=a[s_key],
        scores_b=b[s_key],
        metric_name=metric_name,
        n_boot=n_boot,
        seed=seed,
    )
    out["run_id_a"] = run_id_a
    out["run_id_b"] = run_id_b
    out["split"] = split
    return out


def build_pairwise_significance_table(
    decision_table: pd.DataFrame,
    full_results: pd.DataFrame,
    top_k: int = 2,
) -> pd.DataFrame:
    """Small helper for later extension.

    For each dataset, model, balance mode, and k, compare the top two probes
    by mean test recall at one percent FPR. This function assumes you will
    later join decision rows back to concrete run ids.
    """
    rows = []
    group_cols = ["dataset", "model", "balance_mode", "k"]

    for keys, g in full_results.groupby(group_cols, dropna=False):
        ranked = g.sort_values("test_recall_at_1pct_fpr", ascending=False)
        if len(ranked) < top_k:
            continue
        a = ranked.iloc[0]
        b = ranked.iloc[1]
        rows.append(
            {
                "dataset": keys[0],
                "model": keys[1],
                "balance_mode": keys[2],
                "k": keys[3],
                "probe_a": a["probe"],
                "probe_b": b["probe"],
                "run_id_a": a["run_id"],
                "run_id_b": b["run_id"],
            }
        )

    return pd.DataFrame(rows)
