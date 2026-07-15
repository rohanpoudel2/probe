from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FewShotSelection:
    """A group-aware few-shot sample and its provenance.

    ``k`` is defined as the number of positive scenario groups. In balanced
    mode, every selected scenario contributes exactly one positive and one
    negative example, so the total number of labeled examples is ``2 * k``.
    """

    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    group_ids: np.ndarray


def _validate_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    group_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    groups = np.asarray(group_ids).astype(str)

    if X.ndim != 2:
        raise ValueError(f"X_train must be two-dimensional, got shape {X.shape}")
    if y.ndim != 1 or groups.ndim != 1:
        raise ValueError("y_train and group_ids must be one-dimensional")
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X_train, y_train, and group_ids must have equal lengths")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    labels = set(np.unique(y).tolist())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError(f"Few-shot sampling requires binary labels 0 and 1, got {sorted(labels)}")
    if np.any(np.char.str_len(groups) == 0):
        raise ValueError("group_ids must be non-empty for every training example")
    return X, y.astype(np.int64, copy=False), groups


def _indices_by_group(y: np.ndarray, groups: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
    grouped: dict[str, dict[int, np.ndarray]] = {}
    for group in np.unique(groups):
        group_mask = groups == group
        grouped[group] = {
            0: np.flatnonzero(group_mask & (y == 0)),
            1: np.flatnonzero(group_mask & (y == 1)),
        }
    return grouped


def sample_few_shot_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    seed: int,
    balance_mode: str,
    *,
    group_ids: np.ndarray,
    return_selection: bool = False,
) -> tuple[np.ndarray, np.ndarray] | FewShotSelection:
    """Sample labeled examples without confounding class with scenario.

    Balanced sampling requires groups containing both labels and selects one
    example of each class from each of ``k`` groups. Imbalanced sampling still
    selects positives from distinct groups, but retains all available negative
    examples; it is intended only for explicitly labeled prevalence studies.
    """

    X, y, groups = _validate_inputs(X_train, y_train, group_ids, k)
    rng = np.random.default_rng(seed)
    grouped = _indices_by_group(y, groups)

    if balance_mode == "balanced":
        eligible = np.asarray(
            [group for group, by_label in grouped.items() if len(by_label[0]) and len(by_label[1])],
            dtype=str,
        )
        if k > len(eligible):
            raise ValueError(
                f"Requested k={k} matched scenario groups but only {len(eligible)} contain both labels"
            )
        chosen_groups = rng.choice(eligible, size=k, replace=False)
        selected: list[int] = []
        for group in chosen_groups:
            selected.append(int(rng.choice(grouped[str(group)][1])))
            selected.append(int(rng.choice(grouped[str(group)][0])))
    elif balance_mode == "imbalanced":
        positive_groups = np.asarray(
            [group for group, by_label in grouped.items() if len(by_label[1])],
            dtype=str,
        )
        if k > len(positive_groups):
            raise ValueError(
                f"Requested k={k} positive scenario groups but only {len(positive_groups)} are available"
            )
        chosen_groups = rng.choice(positive_groups, size=k, replace=False)
        selected = [int(rng.choice(grouped[str(group)][1])) for group in chosen_groups]
        selected.extend(np.flatnonzero(y == 0).astype(int).tolist())
    else:
        raise ValueError(f"Unknown balance_mode: {balance_mode}")

    selected_idx = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_idx)
    selection = FewShotSelection(
        X=X[selected_idx],
        y=y[selected_idx],
        indices=selected_idx,
        group_ids=groups[selected_idx],
    )
    if return_selection:
        return selection
    return selection.X, selection.y
