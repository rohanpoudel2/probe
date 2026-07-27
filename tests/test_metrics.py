import numpy as np
import pytest

from evaluation.metrics import (
    alert_rate_summary,
    conformal_alert_p_values,
    compute_brier_score,
    require_disjoint_reference_groups,
    require_independent_reference_groups,
    select_threshold_at_alert_rate,
    wilson_interval,
)


def test_split_conformal_threshold_and_p_values_are_conservative() -> None:
    reference = np.linspace(0.0, 0.9999, 10_000)
    threshold = select_threshold_at_alert_rate(
        reference, max_alert_rate=0.01, min_reference=10_000
    )
    p_values = conformal_alert_p_values(
        reference, np.asarray([threshold, reference.max(), reference.max() + 1.0])
    )
    assert threshold > reference[-100]
    assert p_values[0] <= 0.01
    assert p_values[1] < p_values[0]
    assert p_values[2] == pytest.approx(1 / 10_001)
    assert np.isinf(
        select_threshold_at_alert_rate(
            np.asarray([0.1, 0.2]), max_alert_rate=0.01, min_reference=2
        )
    )


def test_reference_partitions_require_unique_disjoint_groups() -> None:
    calibration = np.asarray([f"cal-{index}" for index in range(100)])
    holdout = np.asarray([f"holdout-{index}" for index in range(100)])
    assert (
        require_independent_reference_groups(
            calibration, min_reference_groups=100
        )
        == 100
    )
    require_disjoint_reference_groups(calibration, holdout)
    with pytest.raises(ValueError, match="group-disjoint"):
        require_disjoint_reference_groups(calibration, np.asarray(["cal-0"]))


def test_holdout_alert_rate_summary_reports_wilson_interval() -> None:
    summary = alert_rate_summary(np.asarray([0.0] * 99 + [1.0]), 0.5)
    assert summary["alerts"] == 1
    assert summary["rate"] == 0.01
    assert summary["ci_low"] < 0.01 < summary["ci_high"]


def test_probability_metrics_reject_raw_scores() -> None:
    with pytest.raises(ValueError, match="probability"):
        compute_brier_score(np.asarray([0, 1]), np.asarray([-2.0, 3.0]))


def test_wilson_interval_bounds_observation() -> None:
    low, high = wilson_interval(10, 100)
    assert 0.0 < low < 0.1 < high < 1.0
