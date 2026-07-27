from __future__ import annotations

import numpy as np
import pandas as pd


SELECTION_KEYS = ["probe", "balance_mode", "layer", "view"]


def _group_keys(df: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    return {
        tuple("<NA>" if pd.isna(value) else str(value) for value in row)
        for row in df[columns].itertuples(index=False, name=None)
    }


def _require_group_coverage(
    expected_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    group_cols: list[str],
    *,
    reason: str,
) -> None:
    missing = sorted(_group_keys(expected_df, group_cols) - _group_keys(observed_df, group_cols))
    if missing:
        raise ValueError(f"{reason}; missing groups={missing[:5]}")


def _finite_metric_rows(
    candidates: pd.DataFrame,
    selection_metric: str,
    group_cols: list[str],
) -> pd.DataFrame:
    finite = candidates.copy()
    finite[selection_metric] = pd.to_numeric(finite[selection_metric], errors="coerce")
    finite = finite[np.isfinite(finite[selection_metric].to_numpy(dtype=float))]
    _require_group_coverage(
        candidates,
        finite,
        group_cols,
        reason=f"No finite {selection_metric} is available for every selection group",
    )
    return finite


def monitor_access_regime(probe: object) -> str:
    """Return the registered access regime encoded by a monitor name."""

    name = str(probe)
    if name.startswith("P"):
        return "white_box"
    if name.startswith("B"):
        return "black_box"
    raise ValueError(
        f"Monitor {name!r} does not use a registered P* (white-box) or B* "
        "(black-box) identifier"
    )


def select_frozen_source_systems(
    summary_df: pd.DataFrame,
    selection_metric: str = "eval_tpr_at_reference_alert_budget_mean",
    selection_k: int | None = None,
) -> pd.DataFrame:
    """Select one layer/view per probe using source eval at a fixed label budget.

    Probe family and label budget are not optimized here. The selected
    layer/view is reused across all ``k`` values and every target task, which
    preserves honest label-efficiency curves.
    """

    if summary_df.empty:
        return pd.DataFrame()
    required = {
        "model",
        "source_task",
        *SELECTION_KEYS,
        "k",
        selection_metric,
    }
    missing = sorted(required.difference(summary_df.columns))
    if missing:
        raise ValueError(f"Source-selection summary lacks columns {missing}")
    available_k = sorted(summary_df["k"].dropna().astype(int).unique().tolist())
    if not available_k:
        raise ValueError("No k values are available for source selection")
    chosen_k = max(available_k) if selection_k is None else int(selection_k)
    group_cols = ["model", "source_task", "probe", "balance_mode"]
    candidates = summary_df[summary_df["k"].astype(int) == chosen_k].copy()
    if candidates.empty:
        raise ValueError(f"selection_k={chosen_k} is not available")
    _require_group_coverage(
        summary_df,
        candidates,
        group_cols,
        reason=f"selection_k={chosen_k} is not available for every selection group",
    )
    candidates = _finite_metric_rows(candidates, selection_metric, group_cols)

    collapsed = (
        candidates.groupby(
            ["model", "source_task", *SELECTION_KEYS], dropna=False
        )[selection_metric]
        .mean()
        .reset_index()
    )
    collapsed = collapsed.sort_values(
        [*group_cols, "layer", "view"],
        kind="mergesort",
    )
    idx = collapsed.groupby(group_cols, dropna=False)[selection_metric].idxmax()
    selected = collapsed.loc[idx].reset_index(drop=True)
    selected["selection_k"] = chosen_k
    selected["selection_metric"] = selection_metric
    selected["selection_rule"] = "source_eval_layer_view_at_fixed_k"
    return selected


def select_primary_source_systems(
    summary_df: pd.DataFrame,
    selection_metric: str = "eval_tpr_at_reference_alert_budget_mean",
    selection_k: int | None = None,
) -> pd.DataFrame:
    """Select one primary monitor per access regime using source data only.

    The fixed selection budget is chosen explicitly, or defaults to the
    largest available ``k``.  Probe family, layer, view, and (where relevant)
    balance mode are selected jointly.  The resulting white-box and black-box
    identities are then reused for every reported label budget and target.
    """

    if summary_df.empty:
        return pd.DataFrame()
    required = {
        "model",
        "source_task",
        *SELECTION_KEYS,
        "k",
        selection_metric,
    }
    missing = sorted(required.difference(summary_df.columns))
    if missing:
        raise ValueError(f"Source-selection summary lacks columns {missing}")
    candidates_all_k = summary_df.copy()
    candidates_all_k["access_regime"] = candidates_all_k["probe"].map(
        monitor_access_regime
    )
    available_k = sorted(candidates_all_k["k"].dropna().astype(int).unique().tolist())
    if not available_k:
        raise ValueError("No k values are available for primary source selection")
    chosen_k = max(available_k) if selection_k is None else int(selection_k)
    group_cols = ["model", "source_task", "access_regime"]
    candidates = candidates_all_k[
        candidates_all_k["k"].astype(int) == chosen_k
    ].copy()
    if candidates.empty:
        raise ValueError(f"selection_k={chosen_k} is not available")
    _require_group_coverage(
        candidates_all_k,
        candidates,
        group_cols,
        reason=f"selection_k={chosen_k} is not available for every access regime",
    )
    candidates = _finite_metric_rows(candidates, selection_metric, group_cols)

    collapsed = (
        candidates.groupby(
            ["model", "source_task", "access_regime", *SELECTION_KEYS],
            dropna=False,
        )[selection_metric]
        .mean()
        .reset_index()
    )

    # Make ties reproducible and conservative: the lexicographically first
    # registered identity wins rather than whichever row happened to be read
    # first from the result directory.
    collapsed = collapsed.sort_values(
        [
            "model",
            "source_task",
            "access_regime",
            "probe",
            "balance_mode",
            "layer",
            "view",
        ],
        kind="mergesort",
    )
    idx = collapsed.groupby(group_cols, dropna=False)[selection_metric].idxmax()
    selected = collapsed.loc[idx].reset_index(drop=True)
    selected["selection_k"] = chosen_k
    selected["selection_metric"] = selection_metric
    selected["selection_rule"] = (
        "source_eval_tpr_at_reference_alert_budget_family_layer_view_at_fixed_k_lexicographic_ties"
    )
    return selected


def apply_frozen_selection(summary_df: pd.DataFrame, selected_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or selected_df.empty:
        return pd.DataFrame()
    merge_cols = ["model", "source_task", *SELECTION_KEYS]
    selection_metadata = ["selection_k", "selection_metric", "selection_rule"]
    if "access_regime" in selected_df.columns:
        selection_metadata.append("access_regime")
    merged = summary_df.merge(
        selected_df[merge_cols + selection_metadata],
        on=merge_cols,
        how="inner",
        validate="many_to_one",
    )
    return merged.sort_values(
        ["model", "source_task", "target_task", "probe", "k"]
    ).reset_index(drop=True)
