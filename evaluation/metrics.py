"""Evaluation metrics: AUROC, Recall@FPR, and the novel FSEI metric."""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under the ROC Curve."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, scores)


def compute_recall_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01
) -> float:
    """Compute recall (TPR) at a given false positive rate threshold.

    Finds the highest recall achievable while keeping FPR <= max_fpr.
    This is the metric that matters most for safety monitoring: how many
    bad examples do you catch without flooding operators with false alarms?
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, scores)

    # Find the highest TPR where FPR <= max_fpr
    valid = fpr <= max_fpr
    if not valid.any():
        return 0.0
    return float(tpr[valid].max())


def compute_fsei(
    recall_by_k: dict[int, float], k_values: list[int]
) -> float:
    """Compute the Few-Shot Efficiency Index (FSEI).

    FSEI is the area under the Recall@1%FPR vs. k curve, normalised to [0, 1]
    by dividing by the area of a perfect detector (100% recall at every k).

    A probe that works well at very low k scores high. A probe that only
    becomes useful at k=50 or 100 scores low, even if its peak is competitive.

    Args:
        recall_by_k: Mapping from k -> mean Recall@1%FPR across seeds.
        k_values: The k grid (must match keys in recall_by_k).

    Returns:
        FSEI score in [0, 1].
    """
    sorted_k = sorted(k_values)
    recalls = np.array([recall_by_k[k] for k in sorted_k])
    k_arr = np.array(sorted_k, dtype=float)

    # Trapezoidal integration
    area = np.trapz(recalls, x=k_arr)

    # Maximum possible area (recall=1.0 everywhere)
    max_area = np.trapz(np.ones_like(k_arr), x=k_arr)

    if max_area == 0:
        return 0.0
    return float(area / max_area)
