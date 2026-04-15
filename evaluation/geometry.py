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


def _safe_cov(X: np.ndarray) -> np.ndarray:
    dim = X.shape[1]
    if len(X) < 2:
        return np.eye(dim, dtype=float) * 1e-8
    cov = np.cov(X, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.asarray([[float(cov)]], dtype=float)
    return np.asarray(cov, dtype=float)


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


def nn_purity(X: np.ndarray, y: np.ndarray) -> float:
    if len(X) < 2:
        return float("nan")
    diffs = X[:, None, :] - X[None, :, :]
    dists = np.sum(diffs * diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    nearest = np.argmin(dists, axis=1)
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

    cov_pos = _safe_cov(pos)
    cov_neg = _safe_cov(neg)
    cov_all = _safe_cov(X)
    eigvals = np.linalg.eigvalsh(cov_all)
    eigvals = np.clip(eigvals, 1e-12, None)
    within_trace = float(np.trace(cov_pos) + np.trace(cov_neg))
    centroid_distance = float(np.linalg.norm(pos.mean(axis=0) - neg.mean(axis=0)))
    return {
        "n_examples": int(len(X)),
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "centroid_distance": centroid_distance,
        "within_class_cov_trace": within_trace,
        "covariance_condition_number": float(eigvals.max() / eigvals.min()),
        "effective_rank": effective_rank(X),
        "anisotropy": float(eigvals.max() / eigvals.mean()),
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
