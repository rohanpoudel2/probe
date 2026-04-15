from __future__ import annotations

import pandas as pd


SYSTEM_COLS = ["probe", "k", "balance_mode", "layer", "view"]


def add_system_name(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["system_name"] = (
        out["probe"].astype(str)
        + "|L" + out["layer"].astype(str)
        + "|" + out["view"].astype(str)
        + "|k" + out["k"].astype(str)
        + "|" + out["balance_mode"].astype(str)
    )
    return out


def select_frozen_source_systems(
    summary_df: pd.DataFrame,
    selection_metric: str = "eval_recall_at_1pct_fpr_mean",
) -> pd.DataFrame:
    """Select one system per model and source task, without conditioning on target task.

    This is the Phase 4 no-leakage selector for transfer. The chosen system is then
    reused for every target task of that source family.
    """
    if summary_df.empty:
        return pd.DataFrame()

    collapsed = (
        summary_df.groupby(["model", "source_task", *SYSTEM_COLS], dropna=False)[selection_metric]
        .mean()
        .reset_index()
    )
    idx = collapsed.groupby(["model", "source_task"], dropna=False)[selection_metric].idxmax()
    selected = collapsed.loc[idx].reset_index(drop=True)
    return selected


def apply_frozen_selection(summary_df: pd.DataFrame, selected_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or selected_df.empty:
        return pd.DataFrame()

    merged = summary_df.merge(
        selected_df[["model", "source_task", *SYSTEM_COLS]],
        on=["model", "source_task", *SYSTEM_COLS],
        how="inner",
    )
    return merged.sort_values(["model", "source_task", "target_task"]).reset_index(drop=True)
