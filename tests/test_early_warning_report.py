import sys

import pandas as pd

from cli.build_early_warning_report import _derive_prefix, _integrate_curve, main


def test_derive_prefix_from_trajectory_view() -> None:
    assert _derive_prefix("trajectory_prefix_p33") == 33
    assert _derive_prefix("trajectory_prefix_stack_p33") == 33
    assert _derive_prefix("trajectory_prompt_end") == 0
    assert _derive_prefix("answer") is None


def test_integrate_curve_with_two_points() -> None:
    rows = pd.DataFrame(
        {
            "prefix_alert_pct": [10, 50, 90],
            "metric": [0.2, 0.6, 1.0],
        }
    )
    value = _integrate_curve(rows, "metric", strategy="uniform")
    assert round(value, 3) == 0.480


def test_report_matches_black_prefixes_across_balance_modes(
    tmp_path, monkeypatch
) -> None:
    rows = []
    for target in ("sycophancy", "motivated_reasoning"):
        for prefix, white_eval, white_transfer in (
            (10, 0.6, 0.6),
            (100, 0.8, 0.8),
        ):
            rows.append(
                {
                    "model": "model",
                    "source_task": "sycophancy",
                    "target_task": target,
                    "view": f"trajectory_prefix_stack_p{prefix}",
                    "probe": "P8_citm",
                    "k": 8,
                    "balance_mode": "balanced",
                    "layer": 5,
                    "eval_tpr_at_reference_alert_budget_mean": white_eval,
                    "transfer_tpr_at_reference_alert_budget_mean": white_transfer,
                }
            )
            rows.append(
                {
                    "model": "model",
                    "source_task": "sycophancy",
                    "target_task": target,
                    "view": f"response_prefix_text_p{prefix}",
                    "probe": "B3_llm_judge_zero_shot",
                    "k": 8,
                    "balance_mode": "none",
                    "layer": -1,
                    "eval_tpr_at_reference_alert_budget_mean": 0.5,
                    "transfer_tpr_at_reference_alert_budget_mean": 0.4,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "task_summary.csv", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_early_warning_report", "--results_dir", str(tmp_path)],
    )
    main()

    report = pd.read_csv(tmp_path / "early_warning_report.csv")
    citm = report[report["probe"] == "P8_citm"]
    assert set(citm["matched_black_balance_mode"]) == {"none"}
    assert citm["matched_black_probe"].notna().all()
    endpoint = pd.read_csv(tmp_path / "early_warning_primary_endpoint.csv")
    assert endpoint.loc[0, "n_model_target_cells"] == 1
    assert pd.notna(endpoint.loc[0, "estimate"])
