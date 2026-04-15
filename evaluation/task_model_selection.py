from __future__ import annotations

import pandas as pd


def rank_models(best_df: pd.DataFrame, metric: str = "transfer_recall_at_1pct_fpr_mean") -> pd.DataFrame:
    if best_df.empty:
        return pd.DataFrame()
    ranking = best_df.copy()
    ranking["ranking_metric"] = ranking[metric].fillna(ranking.get("test_recall_at_1pct_fpr_mean", 0.0))
    ranking = ranking.sort_values(
        ["source_task", "target_task", "ranking_metric"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    ranking["rank_within_task_pair"] = ranking.groupby(["source_task", "target_task"]).cumcount() + 1
    return ranking
