from __future__ import annotations

import numpy as np
import pandas as pd


def add_seed_summary_columns(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """Summarize few-shot-seed variability without claiming population CIs.

    Population uncertainty must be computed from per-example predictions with
    scenario-group resampling. The seed standard deviation here is diagnostic.
    """

    if df.empty:
        return df.copy()
    rows = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: value for column, value in zip(group_cols, group_key)}
        row["uncertainty_unit"] = "few_shot_training_seed_only"
        for metric in metric_cols:
            if metric not in group:
                continue
            values = group[metric].astype(float).to_numpy()
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            row[f"{metric}_seed_std"] = float(np.std(finite)) if len(finite) else float("nan")
            row[f"{metric}_seed_min"] = float(np.min(finite)) if len(finite) else float("nan")
            row[f"{metric}_seed_max"] = float(np.max(finite)) if len(finite) else float("nan")
            row[f"{metric}_n_seeds"] = int(len(finite))
            low_column = f"{metric}_ci_low"
            high_column = f"{metric}_ci_high"
            if low_column in group and high_column in group:
                lows = pd.to_numeric(group[low_column], errors="coerce").to_numpy(
                    dtype=float
                )
                highs = pd.to_numeric(group[high_column], errors="coerce").to_numpy(
                    dtype=float
                )
                finite_lows = lows[np.isfinite(lows)]
                finite_highs = highs[np.isfinite(highs)]
                if len(finite_lows) != len(finite) or len(finite_highs) != len(
                    finite
                ):
                    raise ValueError(
                        f"Metric {metric} has incomplete per-seed confidence intervals"
                    )
                # These are not bootstrap intervals over arbitrary training
                # seeds. Preserve a conservative envelope of the registered
                # per-seed intervals so downstream claim gates cannot turn
                # seed variation into falsely narrow uncertainty.
                row[low_column] = float(np.min(finite_lows))
                row[high_column] = float(np.max(finite_highs))
                row[f"{metric}_ci_aggregation"] = (
                    "conservative_seedwise_interval_envelope"
                )
        rows.append(row)
    return pd.DataFrame(rows)
