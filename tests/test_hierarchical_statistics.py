import numpy as np
import pytest

from evaluation.hierarchical_statistics import (
    hierarchical_paired_curve_aggregate,
    hierarchical_paired_mean_difference,
    hierarchical_paired_rate_difference,
    holm_adjust,
)


def test_hierarchical_difference_resamples_groups_and_seeds() -> None:
    groups = np.tile(np.repeat([f"q{i}" for i in range(6)], 2), 3)
    labels = np.tile(np.tile([0, 1], 6), 3)
    seeds = np.repeat(["0", "1", "2"], 12)
    predictions_a = labels.astype(float)
    predictions_b = np.zeros_like(labels, dtype=float)
    result = hierarchical_paired_rate_difference(
        labels,
        predictions_a,
        predictions_b,
        groups,
        seeds,
        metric="tpr",
        n_boot=100,
        seed=4,
    )
    assert result["mean_diff"] == 1.0
    assert result["ci_low"] == 1.0
    assert result["n_groups"] == 6
    assert result["n_seeds"] == 3


def test_holm_adjustment_preserves_order_and_monotonicity() -> None:
    adjusted = holm_adjust([0.04, 0.01, 0.03])
    assert adjusted == [0.06, 0.03, 0.06]


def test_scenario_groups_receive_equal_weight_despite_repeated_rollouts() -> None:
    # q0 has ten rollouts and q1 has one. Equal group weighting makes method A
    # score 0.5 rather than 10/11.
    labels = np.ones(22, dtype=int)
    groups_one_seed = np.asarray(["q0"] * 10 + ["q1"])
    groups = np.tile(groups_one_seed, 2)
    seeds = np.repeat(["0", "1"], 11)
    predictions_a = np.tile(np.asarray([1.0] * 10 + [0.0]), 2)
    predictions_b = np.zeros(22)
    result = hierarchical_paired_rate_difference(
        labels,
        predictions_a,
        predictions_b,
        groups,
        seeds,
        metric="tpr",
        n_boot=50,
        seed=2,
    )
    assert result["metric_a"] == 0.5
    assert result["mean_diff"] == 0.5


def test_hierarchical_mean_difference_accepts_signed_pair_margins() -> None:
    result = hierarchical_paired_mean_difference(
        np.asarray([0.8, 0.4, 0.8, 0.4]),
        np.asarray([-0.2, -0.4, -0.2, -0.4]),
        np.asarray(["g0", "g1", "g0", "g1"]),
        np.asarray(["0", "0", "1", "1"]),
        n_boot=50,
        seed=1,
    )
    assert result["metric_a"] == 0.6
    assert result["metric_b"] == pytest.approx(-0.3)
    assert result["mean_diff"] == pytest.approx(0.9)


def test_hierarchical_curve_aggregate_resamples_complete_prefix_curves() -> None:
    seeds = np.repeat(["0", "1"], 4)
    groups = np.tile(np.repeat(["g0", "g1"], 2), 2)
    prefixes = np.tile([10, 100], 4)
    cell = {
        "values_a": np.ones(8),
        "values_b": np.zeros(8),
        "group_ids": groups,
        "seed_ids": seeds,
        "prefix_ids": prefixes,
    }
    result = hierarchical_paired_curve_aggregate(
        [cell],
        n_boot=50,
        seed=3,
    )
    assert result["metric_a"] == pytest.approx(1.0)
    assert result["metric_b"] == pytest.approx(0.0)
    assert result["mean_diff"] == pytest.approx(1.0)
    assert result["n_cells"] == 1
