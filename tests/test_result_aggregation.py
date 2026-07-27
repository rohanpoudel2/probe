from __future__ import annotations

import json

import pandas as pd

from evaluation.aggregation import collect_results
from evaluation.task_statistics import add_seed_summary_columns


def test_result_collection_ignores_root_level_prediction_evidence(tmp_path) -> None:
    summary = {
        "run_id": "run-1",
        "probe": "P1_logistic",
        "status": "ok",
        "score": 0.5,
    }
    evidence = {
        "run_id": "run-1",
        "probe": "P1_logistic",
        "example_id": "example-1",
        "predicted_positive": True,
    }
    (tmp_path / "summary.jsonl").write_text(
        json.dumps(summary) + "\n", encoding="utf-8"
    )
    (tmp_path / "falsification_shift_predictions.jsonl").write_text(
        json.dumps(evidence) + "\n", encoding="utf-8"
    )
    collected = collect_results(str(tmp_path))
    assert collected["run_id"].tolist() == ["run-1"]
    assert collected["status"].tolist() == ["ok"]


def test_seed_summary_preserves_conservative_interval_envelope() -> None:
    runs = pd.DataFrame(
        [
            {
                "probe": "P1",
                "seed": 0,
                "reference_holdout_alert_rate": 0.008,
                "reference_holdout_alert_rate_ci_low": 0.006,
                "reference_holdout_alert_rate_ci_high": 0.011,
            },
            {
                "probe": "P1",
                "seed": 1,
                "reference_holdout_alert_rate": 0.010,
                "reference_holdout_alert_rate_ci_low": 0.007,
                "reference_holdout_alert_rate_ci_high": 0.014,
            },
        ]
    )
    summary = add_seed_summary_columns(
        runs,
        group_cols=["probe"],
        metric_cols=["reference_holdout_alert_rate"],
    )
    assert summary.loc[0, "reference_holdout_alert_rate_ci_low"] == 0.006
    assert summary.loc[0, "reference_holdout_alert_rate_ci_high"] == 0.014
    assert (
        summary.loc[0, "reference_holdout_alert_rate_ci_aggregation"]
        == "conservative_seedwise_interval_envelope"
    )
