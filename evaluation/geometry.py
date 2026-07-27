from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


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
        return np.zeros(X.shape[1], dtype=float)
    return normalize_direction(pos.mean(axis=0) - neg.mean(axis=0))


def _covariance_trace(X: np.ndarray) -> float:
    """Return trace(cov(X)) without materializing a hidden_dim² matrix."""

    values = np.asarray(X, dtype=float)
    if len(values) < 2:
        return float(values.shape[1] * 1e-8)
    centered = values - values.mean(axis=0, keepdims=True)
    return float(np.sum(centered * centered) / (len(values) - 1))


def _covariance_spectrum_summary(X: np.ndarray) -> tuple[float, float]:
    """Return condition number and anisotropy from the compact SVD spectrum."""

    values = np.asarray(X, dtype=float)
    dimension = values.shape[1]
    if len(values) < 2:
        return 1.0, 1.0
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = np.square(singular_values) / (len(values) - 1)
    maximum = float(np.max(eigenvalues)) if len(eigenvalues) else 0.0
    # The full covariance has hidden_dim eigenvalues. When rank < hidden_dim,
    # its omitted eigenvalues are exactly zero and receive the same numerical
    # floor used by the previous dense eigendecomposition.
    rank_deficient = len(eigenvalues) < dimension or np.any(eigenvalues <= 1e-12)
    minimum = (
        1e-12
        if rank_deficient
        else max(float(np.min(eigenvalues)), 1e-12)
    )
    mean_eigenvalue = float(np.sum(eigenvalues) / dimension)
    condition = maximum / minimum
    anisotropy = maximum / max(mean_eigenvalue, 1e-12)
    return condition, anisotropy


def effective_rank(X: np.ndarray) -> float:
    if len(X) < 2:
        return 1.0
    centered = X - X.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    total = singular_values.sum()
    if total <= 0:
        return 0.0
    probs = singular_values / total
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(np.exp(entropy))


def nn_purity(
    X: np.ndarray,
    y: np.ndarray,
    *,
    block_size: int = 256,
) -> float:
    if len(X) < 2:
        return float("nan")
    values = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    if values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("Nearest-neighbor purity requires aligned X and y")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    norms = np.sum(values * values, axis=1)
    nearest = np.empty(len(values), dtype=np.int64)
    for start in range(0, len(values), block_size):
        stop = min(start + block_size, len(values))
        distances = (
            norms[start:stop, None]
            + norms[None, :]
            - 2.0 * (values[start:stop] @ values.T)
        )
        np.maximum(distances, 0.0, out=distances)
        local = np.arange(stop - start)
        distances[local, np.arange(start, stop)] = np.inf
        nearest[start:stop] = np.argmin(distances, axis=1)
    return float(np.mean(y[nearest] == y))


def direction_stability(X: np.ndarray, y: np.ndarray, n_boot: int = 16, seed: int = 0) -> float:
    pos = X[y == 1]
    neg = X[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    base = mean_difference_direction(X, y)
    if not np.any(base):
        return float("nan")
    cosines = []
    for _ in range(n_boot):
        pos_sample = pos[rng.integers(0, len(pos), size=len(pos))]
        neg_sample = neg[rng.integers(0, len(neg), size=len(neg))]
        boot = normalize_direction(pos_sample.mean(axis=0) - neg_sample.mean(axis=0))
        if np.any(boot):
            cosines.append(float(np.dot(base, boot)))
    if not cosines:
        return float("nan")
    return float(np.mean(cosines))


def compute_geometry_metrics(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    pos = X[y == 1]
    neg = X[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {
            "n_examples": int(len(X)),
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
            "centroid_distance": float("nan"),
            "within_class_cov_trace": float("nan"),
            "covariance_condition_number": float("nan"),
            "effective_rank": float("nan"),
            "anisotropy": float("nan"),
            "fisher_ratio": float("nan"),
            "nn_purity": float("nan"),
            "direction_stability": float("nan"),
        }

    within_trace = _covariance_trace(pos) + _covariance_trace(neg)
    condition_number, anisotropy = _covariance_spectrum_summary(X)
    centroid_distance = float(np.linalg.norm(pos.mean(axis=0) - neg.mean(axis=0)))
    return {
        "n_examples": int(len(X)),
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "centroid_distance": centroid_distance,
        "within_class_cov_trace": within_trace,
        "covariance_condition_number": condition_number,
        "effective_rank": effective_rank(X),
        "anisotropy": anisotropy,
        "fisher_ratio": float((centroid_distance ** 2) / (within_trace + 1e-12)),
        "nn_purity": nn_purity(X, y),
        "direction_stability": direction_stability(X, y),
    }


def build_direction_alignment(rows: Iterable[dict]) -> pd.DataFrame:
    rows = list(rows)
    out_rows = []
    grouped: dict[tuple[str, int, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["layer"], row["view"], row["split"]), []).append(row)

    for (model, layer, view, split), items in grouped.items():
        for a, b in combinations(sorted(items, key=lambda x: x["task"]), 2):
            direction_a = a["direction"]
            direction_b = b["direction"]
            if direction_a.shape != direction_b.shape:
                continue
            cosine = float(np.dot(direction_a, direction_b))
            out_rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "view": view,
                    "split": split,
                    "task_a": a["task"],
                    "task_b": b["task"],
                    "task_pair_min": min(a["task"], b["task"]),
                    "task_pair_max": max(a["task"], b["task"]),
                    "direction_cosine": cosine,
                    "subspace_overlap": abs(cosine),
                }
            )
    return pd.DataFrame(out_rows)
