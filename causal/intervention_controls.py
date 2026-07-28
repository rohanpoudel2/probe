"""Control utilities for intervention robustness experiments."""

from __future__ import annotations

import numpy as np


def orthogonal_projection(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the projection of ``vector`` onto the columns of ``basis``.

    Least squares handles non-orthogonal and rank-deficient nuisance bases
    without requiring callers to pre-orthogonalize them.
    """
    v = np.asarray(vector, dtype=float)
    B = np.asarray(basis, dtype=float)
    if v.ndim != 1:
        raise ValueError("vector must be 1-D")
    if B.ndim == 1:
        B = B[:, None]
    if B.ndim != 2:
        raise ValueError("basis must be 1-D or 2-D")
    if B.shape[1] == 0:
        return v.copy()
    if B.shape[0] != v.shape[0]:
        raise ValueError("basis shape does not match vector shape")

    coefficients, *_ = np.linalg.lstsq(B, v, rcond=None)
    return v - B @ coefficients


def clamp_trajectory_noise(trajectory: np.ndarray, radius: float) -> np.ndarray:
    """Clamp perturbation magnitude along each step by L2 radius."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    arr = np.asarray(trajectory, dtype=float)
    if arr.ndim not in (2, 3):
        raise ValueError("trajectory must be 2-D or 3-D")
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    safe = np.maximum(norms, 1e-12)
    scale = np.minimum(1.0, radius / safe)
    return arr * scale
