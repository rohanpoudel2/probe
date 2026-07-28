from __future__ import annotations

import json
import sys

import pandas as pd
import pytest
import yaml

from cli import build_final_claim_tables


@pytest.mark.parametrize(
    ("interval_low", "interval_high", "violation", "expected_status", "supported"),
    [
        (0.002, 0.008, False, "supported", True),
        (0.006, 0.014, False, "inconclusive", False),
        (0.020, 0.040, True, "violated", False),
    ],
)
def test_claim_tables_report_empirical_gate_status(
    tmp_path,
    monkeypatch,
    interval_low,
    interval_high,
    violation,
    expected_status,
    supported,
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    run = {
        "run_id": "run-1",
        "probe": "P1_logistic",
        "status": "ok",
        "reference_holdout_alert_rate": (interval_low + interval_high) / 2,
        "reference_holdout_alert_rate_ci_low": interval_low,
        "reference_holdout_alert_rate_ci_high": interval_high,
        "reference_holdout_alert_budget_violation": violation,
    }
    (results_dir / "runs.jsonl").write_text(
        json.dumps(run) + "\n", encoding="utf-8"
    )
    selected = pd.DataFrame(
        [
            {
                "model": "model",
                "source_task": "source",
                "target_task": "target",
                "probe": "P1_logistic",
                "k": 8,
                "transfer_tpr_at_1pct_reference_alert_budget_mean": 0.5,
                "transfer_auroc_mean": 0.7,
                "reference_holdout_alert_rate_mean": (
                    interval_low + interval_high
                )
                / 2,
                "reference_holdout_alert_rate_ci_low": interval_low,
                "reference_holdout_alert_rate_ci_high": interval_high,
            }
        ]
    )
    selected.to_csv(results_dir / "task_cross_task_transfer.csv", index=False)
    selected.iloc[0:0].to_csv(
        results_dir / "task_same_task_calibration.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "comparison_id": "hard-fpr",
                "task_name": "source",
                "metric": "hard_negative_fpr",
                "ci_high": 0.05,
                "metric_a_ci_low": 0.0,
            },
            {
                "comparison_id": "ordering",
                "task_name": "source",
                "metric": "pairwise_order_accuracy",
                "ci_high": 0.0,
                "metric_a_ci_low": 0.4,
            },
        ]
    ).to_csv(results_dir / "falsification_significance.csv", index=False)
    pd.DataFrame(
        [
            {
                "endpoint": "mean_unseen_behavior_auew_uplift",
                "mean_diff": 0.1,
                "ci_low": 0.01,
                "ci_high": 0.2,
                "p_value": 0.02,
                "n_cells": 1,
                "status": "confirmatory_hierarchical_inference_complete",
            }
        ]
    ).to_csv(
        results_dir / "early_warning_primary_inference.csv",
        index=False,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "max_reference_alert_rate": 0.01,
                "claim_gates": {
                    "hard_negative_fpr_noninferiority_margin": 0.01,
                    "pairwise_order_accuracy_floor": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(build_final_claim_tables, "_to_tex", lambda *_: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_final_claim_tables",
            "--results_dir",
            str(results_dir),
            "--config",
            str(config_path),
        ],
    )
    build_final_claim_tables.main()
    main_table = pd.read_csv(results_dir / "claim_main_table.csv")
    gates = pd.read_csv(results_dir / "claim_gate_status.csv")
    assert (
        main_table.loc[0, "endpoint"]
        == "mean_unseen_behavior_auew_uplift"
    )
    assert bool(main_table.loc[0, "claim_supported"])
    reference_gate = gates[
        gates["gate_name"] == "selected_system_reference_alert_budget"
    ].iloc[0]
    assert reference_gate["reference_alert_budget_status"] == expected_status
    assert bool(reference_gate["claim_gate_passed"]) is supported
    hard_gates = gates[gates["gate_name"] == "hard_negative"]
    assert not hard_gates["claim_gate_passed"].astype(bool).any()
