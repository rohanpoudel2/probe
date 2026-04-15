from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means[i] = sample.mean()
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def paired_permutation_pvalue(a: Iterable[float], b: Iterable[float], n_perm: int = 5000, seed: int = 0) -> float:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return float("nan")
    if len(a) == 1:
        return 1.0
    diff = a - b
    obs = abs(diff.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diff))
        stat = abs((diff * signs).mean())
        if stat >= obs:
            count += 1
    return float((count + 1) / (n_perm + 1))


def add_ci_columns(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    grouped = df.groupby(group_cols, dropna=False)
    rows = []
    for group_key, g in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: val for col, val in zip(group_cols, group_key)}
        for metric in metric_cols:
            vals = g[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(vals)) if len(vals) else float("nan")
            row[f"{metric}_std"] = float(np.nanstd(vals)) if len(vals) else float("nan")
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, alpha=alpha)
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
            row[f"{metric}_n"] = int(np.sum(~np.isnan(vals)))
        rows.append(row)
    return pd.DataFrame(rows)
