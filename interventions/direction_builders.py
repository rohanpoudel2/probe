from __future__ import annotations

import numpy as np


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction
    return direction / norm


def mean_difference_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pos = X[y == 1]
    neg = X[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Need at least one positive and one negative example to build a direction.")
    return normalize_direction(pos.mean(axis=0) - neg.mean(axis=0))


def probe_direction(probe, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if getattr(probe, "_direction", None) is not None:
        return normalize_direction(probe._direction)

    clf = getattr(probe, "_clf", None)
    if clf is not None and hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_).reshape(-1)
        scaler = getattr(probe, "_scaler", None)
        if scaler is not None and getattr(scaler, "scale_", None) is not None:
            coef = coef / np.maximum(np.asarray(scaler.scale_), 1e-12)
        if coef.size:
            return normalize_direction(coef)

    centroid_pos = getattr(probe, "_centroid_pos", None)
    centroid_neg = getattr(probe, "_centroid_neg", None)
    if centroid_pos is not None and centroid_neg is not None:
        return normalize_direction(np.asarray(centroid_pos).reshape(-1) - np.asarray(centroid_neg).reshape(-1))

    precision = getattr(probe, "_precision", None)
    if precision is not None:
        pos = X[y == 1]
        neg = X[y == 0]
        if len(pos) and len(neg):
            return normalize_direction(precision @ (pos.mean(axis=0) - neg.mean(axis=0)))

    return mean_difference_direction(X, y)
