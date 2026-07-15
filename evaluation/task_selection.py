from __future__ import annotations

import pandas as pd


SELECTION_KEYS = ["probe", "balance_mode", "layer", "view"]


def add_system_name(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["system_name"] = (
        out["probe"].astype(str)
        + "|L"
        + out["layer"].astype(str)
        + "|"
        + out["view"].astype(str)
        + "|k"
        + out["k"].astype(str)
        + "|"
        + out["balance_mode"].astype(str)
    )
    return out


def select_frozen_source_systems(
    summary_df: pd.DataFrame,
    selection_metric: str = "eval_recall_at_1pct_fpr_mean",
    selection_k: int | None = None,
) -> pd.DataFrame:
    """Select one layer/view per probe using source eval at a fixed label budget.

    Probe family and label budget are not optimized here. The selected
    layer/view is reused across all ``k`` values and every target task, which
    preserves honest label-efficiency curves.
    """

    if summary_df.empty:
        return pd.DataFrame()
    if selection_metric not in summary_df:
        raise ValueError(f"Missing selection metric {selection_metric}")
    available_k = sorted(summary_df["k"].dropna().astype(int).unique().tolist())
    if not available_k:
        raise ValueError("No k values are available for source selection")
    chosen_k = max(available_k) if selection_k is None else int(selection_k)
    candidates = summary_df[summary_df["k"].astype(int) == chosen_k]
    if candidates.empty:
        raise ValueError(f"selection_k={chosen_k} is not available")

    collapsed = (
        candidates.groupby(
            ["model", "source_task", *SELECTION_KEYS], dropna=False
        )[selection_metric]
        .mean()
        .reset_index()
    )
    group_cols = ["model", "source_task", "probe", "balance_mode"]
    idx = collapsed.groupby(group_cols, dropna=False)[selection_metric].idxmax()
    selected = collapsed.loc[idx].reset_index(drop=True)
    selected["selection_k"] = chosen_k
    selected["selection_rule"] = "source_eval_layer_view_at_fixed_k"
    return selected


def apply_frozen_selection(summary_df: pd.DataFrame, selected_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or selected_df.empty:
        return pd.DataFrame()
    merge_cols = ["model", "source_task", *SELECTION_KEYS]
    merged = summary_df.merge(
        selected_df[merge_cols + ["selection_k", "selection_rule"]],
        on=merge_cols,
        how="inner",
        validate="many_to_one",
    )
    return merged.sort_values(
        ["model", "source_task", "target_task", "probe", "k"]
    ).reset_index(drop=True)
