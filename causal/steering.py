"""Directional activation steering helpers."""

from __future__ import annotations

import numpy as np


def steer_activation_trajectory(
    trajectory: np.ndarray,
    direction: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Add a scaled intervention vector to each trajectory step."""
    if not isinstance(alpha, (int, float)) or alpha < -10 or alpha > 10:
        raise ValueError("alpha must be a real scalar in [-10, 10]")
    arr = np.asarray(trajectory, dtype=float)
    vec = np.asarray(direction, dtype=float)
    if arr.ndim not in (2, 3):
        raise ValueError("trajectory must be 2-D or 3-D")
    if vec.ndim != 1:
        raise ValueError("direction must be a 1-D vector")
    if vec.shape[0] != arr.shape[-1]:
        raise ValueError("direction shape does not match trajectory feature dimension")

    return arr + alpha * vec
