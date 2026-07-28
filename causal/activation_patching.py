"""Activation-level patching controls for causal sanity checks."""

from __future__ import annotations

import numpy as np


def patch_activation_trajectory(
    trajectory: np.ndarray,
    patch: np.ndarray,
    start: int = 0,
    end: int | None = None,
) -> np.ndarray:
    """Replace a prefix segment of a trajectory with a fixed patch tensor.

    Supports 2-D (steps, dim) trajectories and 3-D batched inputs by applying
    the same patch slice across the batch axis.
    """
    arr = np.asarray(trajectory, dtype=float)
    if arr.ndim not in (2, 3):
        raise ValueError("trajectory must be (steps, dim) or (batch, steps, dim)")
    patch_arr = np.asarray(patch, dtype=float)
    if patch_arr.ndim != 2:
        raise ValueError("patch must be (steps, dim)")

    if patch_arr.shape[1] != arr.shape[-1]:
        raise ValueError("patch feature dimension must match trajectory")
    if end is None:
        end = start + patch_arr.shape[0]
    if start < 0 or end <= start or end > arr.shape[-2]:
        raise ValueError("invalid patch interval")
    if end - start != patch_arr.shape[0]:
        raise ValueError("patch length must match the selected interval")

    patched = np.array(arr, copy=True)
    if patched.ndim == 2:
        patched[start:end, :] = patch_arr
    else:
        patched[:, start:end, :] = patch_arr[None, ...]
    return patched
