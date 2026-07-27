from __future__ import annotations

import numpy as np

from evaluation.geometry import compute_geometry_metrics, nn_purity


def test_blockwise_nn_purity_matches_dense_reference() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(37, 11))
    labels = np.asarray([0, 1] * 18 + [0])
    differences = values[:, None, :] - values[None, :, :]
    distances = np.sum(differences * differences, axis=-1)
    np.fill_diagonal(distances, np.inf)
    expected = float(
        np.mean(labels[np.argmin(distances, axis=1)] == labels)
    )
    assert nn_purity(values, labels, block_size=8) == expected


def test_geometry_metrics_remain_finite_in_high_dimension() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(40, 512))
    labels = np.asarray([0] * 20 + [1] * 20)
    metrics = compute_geometry_metrics(values, labels)
    for name in (
        "centroid_distance",
        "within_class_cov_trace",
        "covariance_condition_number",
        "effective_rank",
        "anisotropy",
        "fisher_ratio",
        "nn_purity",
        "direction_stability",
    ):
        assert np.isfinite(metrics[name])
