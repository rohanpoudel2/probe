"""Evaluation metrics and significance helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under the ROC Curve."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, scores)


def compute_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, scores)


def compute_recall_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01
) -> float:
    """Compute recall at a fixed false-positive-rate threshold."""
    if len(np.unique(y_true)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= max_fpr
    if not valid.any():
        return 0.0
    return float(tpr[valid].max())


def compute_brier_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if len(scores) == 0:
        return float("nan")
    scores = np.clip(scores, 0.0, 1.0)
    return float(np.mean((scores - y_true) ** 2))


def compute_ece(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if len(scores) == 0:
        return float("nan")
    scores = np.clip(scores, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_ids = np.digitize(scores, bins[1:-1], right=True)
    ece = 0.0
    for idx in range(n_bins):
        mask = bucket_ids == idx
        if not np.any(mask):
            continue
        conf = float(scores[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (np.sum(mask) / len(scores)) * abs(acc - conf)
    return float(ece)


def compute_fsei(metric_by_k: dict[int, float], k_values: list[int], weighting: str = "inverse_k") -> float:
    available = [(int(k), float(metric_by_k[k])) for k in sorted(k_values) if k in metric_by_k and not np.isnan(metric_by_k[k])]
    if not available:
        return float("nan")
    ks = np.asarray([k for k, _ in available], dtype=float)
    vals = np.asarray([v for _, v in available], dtype=float)
    if weighting == "inverse_k":
        weights = 1.0 / np.maximum(ks, 1.0)
    else:
        weights = np.ones_like(ks)
    weights = weights / weights.sum()
    return float(np.sum(weights * vals))


def paired_bootstrap_metric_diff(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap for comparing two scoring functions on the same labels."""
    if not (len(y_true) == len(scores_a) == len(scores_b)):
        raise ValueError("y_true, scores_a, and scores_b must have the same length")

    rng = np.random.default_rng(seed)
    n = len(y_true)

    observed_a = float(metric_fn(y_true, scores_a))
    observed_b = float(metric_fn(y_true, scores_b))
    observed_diff = observed_a - observed_b

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        diff = float(metric_fn(sample_y, scores_a[idx]) - metric_fn(sample_y, scores_b[idx]))
        diffs.append(diff)

    if not diffs:
        return {
            "mean_diff": observed_diff,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }

    diffs = np.array(diffs)
    sign_flip = min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_value = float(min(1.0, 2 * sign_flip))
    return {
        "mean_diff": observed_diff,
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_value": p_value,
    }
