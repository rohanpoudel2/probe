"""Evaluation metrics for reference-calibrated operational monitoring."""

from __future__ import annotations

from math import sqrt

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


def select_threshold_at_alert_rate(
    reference_scores: np.ndarray,
    max_alert_rate: float = 0.01,
    *,
    min_reference: int = 100,
) -> float:
    """Fit a conservative split-conformal alert threshold without labels.

    For a future score ``s``, the conformal p-value is
    ``(1 + count(reference_scores >= s)) / (n + 1)``. Predictions use
    ``score >= threshold``. Ties at the boundary are excluded, so the
    resulting decision is conservative.
    """

    values = np.asarray(reference_scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("reference_scores must be a finite one-dimensional array")
    if not 0.0 < max_alert_rate < 1.0:
        raise ValueError(
            f"max_alert_rate must be between 0 and 1, got {max_alert_rate}"
        )
    if len(values) < min_reference:
        raise ValueError(
            f"Frozen {max_alert_rate:.3%} alert calibration requires at least "
            f"{min_reference} reference examples; found {len(values)}"
        )

    # A future point is alertable only when at most ``max_rank`` calibration
    # scores are greater than or equal to it. Moving one ULP above the boundary
    # makes tied calibration scores fail closed.
    max_rank = int(np.floor(max_alert_rate * (len(values) + 1))) - 1
    if max_rank < 0:
        # The minimum attainable conformal p-value is 1 / (n + 1), so no
        # finite future score is alertable at a smaller requested rate.
        return float("inf")
    descending = np.sort(values)[::-1]
    boundary = float(descending[min(max_rank, len(values) - 1)])
    return float(np.nextafter(boundary, np.inf))


def conformal_alert_p_values(
    reference_scores: np.ndarray, test_scores: np.ndarray
) -> np.ndarray:
    """Return conservative upper-tail conformal p-values for arbitrary scores."""

    reference = np.asarray(reference_scores, dtype=float)
    test = np.asarray(test_scores, dtype=float)
    if reference.ndim != 1 or test.ndim != 1:
        raise ValueError("Reference and test scores must be one-dimensional")
    if len(reference) == 0:
        raise ValueError("Reference scores cannot be empty")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(test)):
        raise ValueError("Conformal scores must be finite")
    sorted_reference = np.sort(reference)
    # searchsorted(left) gives count(reference < score); subtracting from n
    # yields count(reference >= score), including ties conservatively.
    counts_ge = len(reference) - np.searchsorted(
        sorted_reference, test, side="left"
    )
    return (1.0 + counts_ge.astype(float)) / (len(reference) + 1.0)


def require_independent_reference_groups(
    group_ids: np.ndarray,
    *,
    min_reference_groups: int,
) -> int:
    """Require one calibration observation per independent reference group."""

    groups = np.asarray(group_ids).astype(str)
    if groups.ndim != 1:
        raise ValueError("Reference group_ids must be a one-dimensional vector")
    if min_reference_groups < 1:
        raise ValueError("min_reference_groups must be positive")
    unique_groups = np.unique(groups)
    if np.any(np.char.str_len(unique_groups) == 0):
        raise ValueError("Reference group IDs must be non-empty")
    if len(unique_groups) != len(groups):
        raise ValueError(
            "Alert calibration requires exactly one observation per independent "
            f"reference group; found {len(groups)} rows from {len(unique_groups)} groups"
        )
    if len(unique_groups) < min_reference_groups:
        raise ValueError(
            f"Alert calibration requires at least {min_reference_groups} independent "
            f"reference groups; found {len(unique_groups)}"
        )
    return int(len(unique_groups))


def require_disjoint_reference_groups(
    calibration_group_ids: np.ndarray,
    holdout_group_ids: np.ndarray,
) -> None:
    """Reject reference partitions that share an underlying scenario group."""

    calibration = np.asarray(calibration_group_ids).astype(str)
    holdout = np.asarray(holdout_group_ids).astype(str)
    if calibration.ndim != 1 or holdout.ndim != 1:
        raise ValueError("Reference group IDs must be one-dimensional vectors")
    overlap = np.intersect1d(calibration, holdout)
    if len(overlap):
        preview = ", ".join(overlap[:5].tolist())
        raise ValueError(
            "Reference calibration and holdout partitions must be group-disjoint; "
            f"found {len(overlap)} overlapping groups (for example: {preview})"
        )


def compute_alert_rate(scores: np.ndarray, threshold: float) -> float:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite one-dimensional array")
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values >= float(threshold)))


def alert_rate_summary(scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Return an alert count, rate, and two-sided 95% Wilson interval."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite one-dimensional array")
    if len(values) == 0:
        raise ValueError("Alert-rate evaluation requires at least one score")
    alerts = int(np.sum(values >= float(threshold)))
    low, high = wilson_interval(alerts, len(values))
    return {
        "alerts": alerts,
        "total": int(len(values)),
        "rate": float(alerts / len(values)),
        "ci_low": low,
        "ci_high": high,
    }


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
    results must use :func:`select_threshold_at_alert_rate` on a separate
    reference partition followed by :func:`compute_recall_at_threshold`.
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
