from __future__ import annotations

import pandas as pd

from evaluation.metrics import compute_fsei


def select_best_view_layer(
    summary: pd.DataFrame,
    selection_metric: str = "eval_tpr_at_1pct_reference_alert_budget_mean",
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    idx = summary.groupby(
        ["probe", "k", "balance_mode", "source_task", "target_task", "model"],
        dropna=False,
    )[selection_metric].idxmax()
    return summary.loc[idx].reset_index(drop=True)


def make_view_layer_table(best_df: pd.DataFrame) -> pd.DataFrame:
    if best_df.empty:
        return pd.DataFrame()
    cols = [
        "probe",
        "k",
        "balance_mode",
        "source_task",
        "target_task",
        "model",
        "layer",
        "view",
        "eval_tpr_at_1pct_reference_alert_budget_mean",
        "eval_tpr_at_1pct_reference_alert_budget_ci_low",
        "eval_tpr_at_1pct_reference_alert_budget_ci_high",
        "test_tpr_at_1pct_reference_alert_budget_mean",
        "test_tpr_at_1pct_reference_alert_budget_ci_low",
        "test_tpr_at_1pct_reference_alert_budget_ci_high",
        "transfer_tpr_at_1pct_reference_alert_budget_mean",
        "transfer_tpr_at_1pct_reference_alert_budget_ci_low",
        "transfer_tpr_at_1pct_reference_alert_budget_ci_high",
        "reference_holdout_alert_rate_mean",
        "reference_holdout_alert_rate_ci_low",
        "reference_holdout_alert_rate_ci_high",
        "test_auroc_mean",
        "transfer_auroc_mean",
    ]
    cols = [c for c in cols if c in best_df.columns]
    return best_df[cols].sort_values(["source_task", "target_task", "model", "probe", "k"]).reset_index(drop=True)


def make_transfer_table(best_df: pd.DataFrame) -> pd.DataFrame:
    if best_df.empty:
        return pd.DataFrame()
    cols = [
        "probe",
        "k",
        "balance_mode",
        "source_task",
        "target_task",
        "model",
        "layer",
        "view",
        "transfer_auroc_mean",
        "transfer_auroc_ci_low",
        "transfer_auroc_ci_high",
        "transfer_tpr_at_1pct_reference_alert_budget_mean",
        "transfer_tpr_at_1pct_reference_alert_budget_ci_low",
        "transfer_tpr_at_1pct_reference_alert_budget_ci_high",
        "test_auroc_mean",
        "test_tpr_at_1pct_reference_alert_budget_mean",
        "reference_holdout_alert_rate_mean",
        "reference_holdout_alert_rate_ci_low",
        "reference_holdout_alert_rate_ci_high",
    ]
    cols = [c for c in cols if c in best_df.columns]
    return best_df[cols].sort_values(["source_task", "target_task", "model", "probe", "k"]).reset_index(drop=True)


def compute_task_fsei(
    best_df: pd.DataFrame,
    outcome_metric: str = "transfer_tpr_at_1pct_reference_alert_budget_mean",
) -> pd.DataFrame:
    if best_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["probe", "balance_mode", "source_task", "target_task", "model"]
    for keys, group in best_df.groupby(group_cols, dropna=False):
        metric_by_k = {}
        fallback_by_k = {}
        for _, row in group.iterrows():
            k = int(row["k"])
            primary = row.get(outcome_metric, float("nan"))
            fallback = row.get(
                "test_tpr_at_1pct_reference_alert_budget_mean", float("nan")
            )
            metric_by_k[k] = float(primary) if pd.notnull(primary) else float("nan")
            fallback_by_k[k] = float(fallback) if pd.notnull(fallback) else float("nan")

        chosen = {
            k: (metric_by_k[k] if not pd.isna(metric_by_k[k]) else fallback_by_k.get(k, float("nan")))
            for k in set(metric_by_k) | set(fallback_by_k)
        }
        valid_k = sorted(k for k, value in chosen.items() if not pd.isna(value))
        rows.append(
            {
                "probe": keys[0],
                "balance_mode": keys[1],
                "source_task": keys[2],
                "target_task": keys[3],
                "model": keys[4],
                "fsei_metric": outcome_metric,
                "task_fsei": compute_fsei(chosen, valid_k, weighting="inverse_k") if valid_k else float("nan"),
                "k_min": min(valid_k) if valid_k else float("nan"),
                "k_max": max(valid_k) if valid_k else float("nan"),
                "n_k": len(valid_k),
            }
        )
    return pd.DataFrame(rows)
