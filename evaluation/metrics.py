"""Evaluation metrics for frozen operating-point monitoring."""

from __future__ import annotations

from math import sqrt
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _as_aligned_arrays(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    if y.ndim != 1 or values.ndim != 1 or len(y) != len(values):
        raise ValueError("y_true and scores must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    return y, values


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y, values = _as_aligned_arrays(y_true, scores)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, values))


def compute_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y, values = _as_aligned_arrays(y_true, scores)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, values))


def select_threshold_at_fpr(
    y_calibration: np.ndarray,
    scores: np.ndarray,
    max_fpr: float = 0.01,
    *,
    min_negatives: int = 100,
) -> float:
    """Select the most permissive threshold meeting an empirical FPR bound.

    Only negative calibration examples affect the threshold. Ties are handled
    conservatively and predictions use ``score >= threshold``. This function
    must never be called on a final evaluation or transfer target.
    """

    y, values = _as_aligned_arrays(y_calibration, scores)
    if not 0.0 < max_fpr < 1.0:
        raise ValueError(f"max_fpr must be between 0 and 1, got {max_fpr}")
    negatives = values[y == 0]
    if len(negatives) < min_negatives:
        raise ValueError(
            f"Frozen {max_fpr:.3%} FPR calibration requires at least {min_negatives} "
            f"negative examples; found {len(negatives)}"
        )

    # A threshold immediately above the maximum always yields zero false
    # positives. Each unique score is then considered from strict to permissive.
    candidates = np.concatenate(
        ([np.nextafter(float(negatives.max()), np.inf)], np.unique(negatives)[::-1])
    )
    valid = [
        float(threshold)
        for threshold in candidates
        if float(np.mean(negatives >= threshold)) <= max_fpr
    ]
    if not valid:  # Defensive: the nextafter candidate should always be valid.
        raise RuntimeError("Could not construct a threshold satisfying the requested FPR")
    return min(valid)


def require_independent_calibration_negatives(
    y_calibration: np.ndarray,
    group_ids: np.ndarray,
    *,
    min_negative_groups: int,
) -> int:
    """Reject tail calibration that inflates sample size with repeated rollouts."""

    y = np.asarray(y_calibration)
    groups = np.asarray(group_ids).astype(str)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups):
        raise ValueError("Calibration labels and group_ids must be aligned vectors")
    if min_negative_groups < 1:
        raise ValueError("min_negative_groups must be positive")
    negative_groups = np.unique(groups[y == 0])
    if np.any(np.char.str_len(negative_groups) == 0):
        raise ValueError("Calibration group IDs must be non-empty")
    n_negative_rows = int(np.sum(y == 0))
    if len(negative_groups) != n_negative_rows:
        raise ValueError(
            "Frozen threshold calibration requires one negative observation per independent "
            f"scenario group; found {n_negative_rows} rows from {len(negative_groups)} groups"
        )
    if len(negative_groups) < min_negative_groups:
        raise ValueError(
            f"Frozen threshold calibration requires at least {min_negative_groups} "
            f"independent negative scenario groups; found {len(negative_groups)}"
        )
    return int(len(negative_groups))


def compute_fpr_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    y, values = _as_aligned_arrays(y_true, scores)
    negatives = values[y == 0]
    if len(negatives) == 0:
        return float("nan")
    return float(np.mean(negatives >= threshold))


def compute_recall_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    y, values = _as_aligned_arrays(y_true, scores)
    positives = values[y == 1]
    if len(positives) == 0:
        return float("nan")
    return float(np.mean(positives >= threshold))


def compute_recall_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01
) -> float:
    """Compute an oracle ROC summary using labels from the evaluated split.

    This is retained for exploratory plots only. Confirmatory claims and transfer
    results must use :func:`select_threshold_at_fpr` on a separate calibration
    split followed by :func:`compute_recall_at_threshold`.
    """

    y, values = _as_aligned_arrays(y_true, scores)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, values)
    valid = fpr <= max_fpr
    return float(tpr[valid].max()) if valid.any() else 0.0


def _validate_probabilities(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Calibration metrics require finite probability scores in [0, 1]")
    return values


def compute_brier_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    values = _validate_probabilities(scores)
    if len(y) != len(values):
        raise ValueError("y_true and scores must have equal lengths")
    if len(values) == 0:
        return float("nan")
    return float(np.mean((values - y) ** 2))


def compute_ece(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    values = _validate_probabilities(scores)
    if len(y) != len(values):
        raise ValueError("y_true and scores must have equal lengths")
    if len(values) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_ids = np.digitize(values, bins[1:-1], right=True)
    ece = 0.0
    for idx in range(n_bins):
        mask = bucket_ids == idx
        if not np.any(mask):
            continue
        confidence = float(values[mask].mean())
        accuracy = float(y[mask].mean())
        ece += (np.sum(mask) / len(values)) * abs(accuracy - confidence)
    return float(ece)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for an observed FPR or recall proportion."""

    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("successes and total must satisfy 0 <= successes <= total and total > 0")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def compute_fsei(metric_by_k: dict[int, float], k_values: list[int], weighting: str = "inverse_k") -> float:
    available = [
        (int(k), float(metric_by_k[k]))
        for k in sorted(k_values)
        if k in metric_by_k and not np.isnan(metric_by_k[k])
    ]
    if not available:
        return float("nan")
    ks = np.asarray([k for k, _ in available], dtype=float)
    vals = np.asarray([v for _, v in available], dtype=float)
    weights = 1.0 / np.maximum(ks, 1.0) if weighting == "inverse_k" else np.ones_like(ks)
    weights = weights / weights.sum()
    return float(np.sum(weights * vals))


def paired_group_bootstrap_metric_diff(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    group_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired cluster bootstrap over independent scenario groups."""

    y, a = _as_aligned_arrays(y_true, scores_a)
    _, b = _as_aligned_arrays(y_true, scores_b)
    groups = np.asarray(group_ids).astype(str)
    if len(groups) != len(y):
        raise ValueError("group_ids must align with y_true")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Grouped bootstrap requires at least two independent groups")

    # Precompute each group's row indices once. The resampling draws and their
    # order are unchanged, so results are identical to rescanning ``groups``
    # every iteration, but the per-iteration cost drops from O(n_boot * G * N)
    # array scans to a dictionary lookup plus concatenation.
    group_to_indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    observed_diff = float(metric_fn(y, a) - metric_fn(y, b))
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_to_indices[group] for group in sampled_groups])
        sample_y = y[sampled_indices]
        if len(np.unique(sample_y)) < 2:
            continue
        diffs.append(float(metric_fn(sample_y, a[sampled_indices]) - metric_fn(sample_y, b[sampled_indices])))

    if not diffs:
        return {"mean_diff": observed_diff, "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}
    values = np.asarray(diffs)
    sign_flip = min(float(np.mean(values <= 0.0)), float(np.mean(values >= 0.0)))
    return {
        "mean_diff": observed_diff,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_value": min(1.0, 2.0 * sign_flip),
    }


def paired_bootstrap_metric_diff(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Deprecated row bootstrap retained for compatibility.

    New research code should call :func:`paired_group_bootstrap_metric_diff`.
    """

    row_groups = np.arange(len(y_true)).astype(str)
    return paired_group_bootstrap_metric_diff(
        y_true, scores_a, scores_b, row_groups, metric_fn, n_boot=n_boot, seed=seed
    )
