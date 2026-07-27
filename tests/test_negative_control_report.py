from __future__ import annotations

import pandas as pd

from cli.build_negative_control_report import _mean_metric


def test_negative_control_summary_excludes_unchanged_black_box_rows(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "access_regime": "white_box",
                "transfer_tpr_at_1pct_reference_alert_budget_mean": 0.7,
            },
            {
                "access_regime": "black_box",
                "transfer_tpr_at_1pct_reference_alert_budget_mean": 0.1,
            },
        ]
    ).to_csv(tmp_path / "task_cross_task_transfer.csv", index=False)
    assert (
        _mean_metric(
            tmp_path,
            "transfer_tpr_at_1pct_reference_alert_budget_mean",
        )
        == 0.7
    )
