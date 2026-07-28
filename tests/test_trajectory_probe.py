from __future__ import annotations

import numpy as np
import pytest

from probes.trajectory import CITMProbe


def test_citm_probe_refuses_missing_step_metadata() -> None:
    probe = CITMProbe()
    with pytest.raises(ValueError, match="trajectory_steps"):
        probe.fit(np.ones((4, 6)), np.array([0, 1, 0, 1]))


def test_citm_probe_fits_and_scores_with_expected_step_count() -> None:
    # Two trajectory steps, each with three-dimensional per-step features.
    X = np.array(
        [
            [1.0, 0.0, 0.0, 1.1, 0.1, 0.2],
            [0.0, 1.0, 0.0, 0.1, 1.0, 0.2],
            [1.0, 1.0, 0.0, 1.2, 1.1, 0.2],
            [0.0, 0.0, 1.0, 0.2, 0.1, 0.9],
        ],
        dtype=float,
    )
    y = np.array([1, 0, 1, 0], dtype=int)
    probe = CITMProbe(trajectory_steps=2)
    probe.fit(X, y)
    scores = probe.score(X)
    assert scores.shape == (len(X),)
    assert np.all(np.isfinite(scores))
    assert np.all((0.0 <= scores) & (scores <= 1.0))
