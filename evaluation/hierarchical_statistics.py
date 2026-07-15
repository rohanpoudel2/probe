from __future__ import annotations

from typing import Literal

import numpy as np


RateMetric = Literal["tpr", "fpr", "positive_rate"]


def _metric_mask(labels: np.ndarray, metric: RateMetric) -> np.ndarray:
    if metric == "tpr":
        return labels == 1
    if metric == "fpr":
        return labels == 0
    if metric == "positive_rate":
        return np.ones(len(labels), dtype=bool)
    raise ValueError(f"Unsupported rate metric: {metric}")


def hierarchical_paired_mean_difference(
    values_a: np.ndarray,
    values_b: np.ndarray,
    group_ids: np.ndarray,
    seed_ids: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap a paired mean difference over seeds and scenario groups."""

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    groups = np.asarray(group_ids).astype(str)
    seeds = np.asarray(seed_ids).astype(str)
    if not (a.ndim == b.ndim == groups.ndim == seeds.ndim == 1):
        raise ValueError("All inputs must be one-dimensional")
    if len({len(a), len(b), len(groups), len(seeds)}) != 1 or not len(a):
        raise ValueError("All inputs must have equal non-zero lengths")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Paired values must be finite")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be positive")

    unique_groups = np.unique(groups)
    unique_seeds = np.unique(seeds)
    if len(unique_groups) < 2 or len(unique_seeds) < 2:
        raise ValueError(
            "Hierarchical inference requires at least two groups and two seeds"
        )

    cell_a: dict[tuple[str, str], float] = {}
    cell_b: dict[tuple[str, str], float] = {}
    for seed_id in unique_seeds:
        for group_id in unique_groups:
            indices = np.flatnonzero((seeds == seed_id) & (groups == group_id))
            if not len(indices):
                raise ValueError(
                    f"Incomplete seed-by-group prediction grid at seed={seed_id}, group={group_id}"
                )
            cell_a[(seed_id, group_id)] = float(np.mean(a[indices]))
            cell_b[(seed_id, group_id)] = float(np.mean(b[indices]))

    # Equal weight per seed-by-scenario cell prevents groups with more repeated
    # rollouts from masquerading as additional independent evidence.
    observed_a = float(np.mean(list(cell_a.values())))
    observed_b = float(np.mean(list(cell_b.values())))
    observed_diff = observed_a - observed_b
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        sampled_seeds = rng.choice(unique_seeds, size=len(unique_seeds), replace=True)
        sampled_groups = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        sample_a: list[float] = []
        sample_b: list[float] = []
        for sampled_seed in sampled_seeds:
            for sampled_group in sampled_groups:
                key = (str(sampled_seed), str(sampled_group))
                sample_a.append(cell_a[key])
                sample_b.append(cell_b[key])
        diffs[iteration] = float(np.mean(sample_a) - np.mean(sample_b))

    sign_flip = min(float(np.mean(diffs <= 0.0)), float(np.mean(diffs >= 0.0)))
    return {
        "metric_a": observed_a,
        "metric_b": observed_b,
        "mean_diff": observed_diff,
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_value": min(1.0, 2.0 * sign_flip),
        "n_groups": int(len(unique_groups)),
        "n_seeds": int(len(unique_seeds)),
        "n_rows": int(len(a)),
    }


def hierarchical_paired_rate_difference(
    labels: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    group_ids: np.ndarray,
    seed_ids: np.ndarray,
    *,
    metric: RateMetric = "tpr",
    n_boot: int = 5000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap a paired monitor difference over seeds and scenario groups."""

    labels = np.asarray(labels, dtype=np.int64)
    a = np.asarray(predictions_a, dtype=float)
    b = np.asarray(predictions_b, dtype=float)
    groups = np.asarray(group_ids).astype(str)
    seeds = np.asarray(seed_ids).astype(str)
    if not (labels.ndim == a.ndim == b.ndim == groups.ndim == seeds.ndim == 1):
        raise ValueError("All inputs must be one-dimensional")
    if len({len(labels), len(a), len(b), len(groups), len(seeds)}) != 1:
        raise ValueError("All inputs must have equal lengths")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if np.any((a < 0.0) | (a > 1.0) | (b < 0.0) | (b > 1.0)):
        raise ValueError("Predictions must be binary or probabilities in [0, 1]")

    mask = _metric_mask(labels, metric)
    if not np.any(mask):
        raise ValueError(f"No examples are eligible for metric {metric}")
    return hierarchical_paired_mean_difference(
        a[mask],
        b[mask],
        groups[mask],
        seeds[mask],
        n_boot=n_boot,
        seed=seed,
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise error correction preserving input order."""

    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    if np.any((values < 0.0) | (values > 1.0) | ~np.isfinite(values)):
        raise ValueError("p-values must be finite values in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()
