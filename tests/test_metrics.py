import numpy as np
import pytest

from evaluation.metrics import (
    compute_brier_score,
    compute_fpr_at_threshold,
    compute_recall_at_threshold,
    paired_group_bootstrap_metric_diff,
    require_independent_calibration_negatives,
    select_threshold_at_fpr,
    wilson_interval,
)


def test_frozen_threshold_uses_negative_calibration_scores_only() -> None:
    labels = np.asarray([0] * 100 + [1] * 3)
    negative_scores = np.linspace(0.0, 0.99, 100)
    scores = np.concatenate([negative_scores, np.asarray([-100.0, 50.0, 100.0])])
    threshold = select_threshold_at_fpr(labels, scores, max_fpr=0.01, min_negatives=100)
    assert compute_fpr_at_threshold(labels, scores, threshold) <= 0.01
    changed_positive_scores = np.concatenate([negative_scores, np.asarray([1e9, 1e9, 1e9])])
    assert select_threshold_at_fpr(
        labels, changed_positive_scores, max_fpr=0.01, min_negatives=100
    ) == threshold


def test_threshold_ties_are_conservative() -> None:
    labels = np.zeros(100, dtype=int)
    scores = np.asarray([1.0] * 2 + [0.0] * 98)
    threshold = select_threshold_at_fpr(labels, scores, max_fpr=0.01, min_negatives=100)
    assert threshold > 1.0
    assert compute_fpr_at_threshold(labels, scores, threshold) == 0.0


def test_repeated_rollouts_do_not_inflate_calibration_sample_size() -> None:
    labels = np.zeros(100, dtype=int)
    repeated_groups = np.repeat([f"q{i}" for i in range(10)], 10)
    with pytest.raises(ValueError, match="one negative observation per independent"):
        require_independent_calibration_negatives(
            labels, repeated_groups, min_negative_groups=100
        )
    assert (
        require_independent_calibration_negatives(
            labels, np.asarray([f"q{i}" for i in range(100)]), min_negative_groups=100
        )
        == 100
    )


def test_probability_metrics_reject_raw_scores() -> None:
    with pytest.raises(ValueError, match="probability"):
        compute_brier_score(np.asarray([0, 1]), np.asarray([-2.0, 3.0]))


def test_grouped_bootstrap_preserves_pairing() -> None:
    labels = np.tile([0, 1], 8)
    groups = np.repeat([f"q{i}" for i in range(8)], 2)
    good = labels.astype(float)
    bad = 1.0 - good
    result = paired_group_bootstrap_metric_diff(
        labels,
        good,
        bad,
        groups,
        lambda y, scores: compute_recall_at_threshold(y, scores, 0.5),
        n_boot=100,
        seed=3,
    )
    assert result["mean_diff"] == 1.0
    assert result["ci_low"] == 1.0


def test_wilson_interval_bounds_observation() -> None:
    low, high = wilson_interval(10, 100)
    assert 0.0 < low < 0.1 < high < 1.0
