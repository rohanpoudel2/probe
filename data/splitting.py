"""Reproducible dataset splitting and few shot sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import Dataset


def make_splits(
    dataset: Dataset,
    seed: int = 42,
    test_frac: float = 0.15,
    eval_frac: float = 0.15,
) -> dict[str, np.ndarray]:
    """Split dataset into train_pool, eval, and test.

    Test and eval are class balanced. Train pool contains the remaining data.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset["label"])

    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    n_test_per_class = int(min(len(pos_indices), len(neg_indices)) * test_frac)
    test_pos = pos_indices[:n_test_per_class]
    test_neg = neg_indices[:n_test_per_class]
    test_indices = np.concatenate([test_pos, test_neg])
    rng.shuffle(test_indices)

    remaining_pos = pos_indices[n_test_per_class:]
    remaining_neg = neg_indices[n_test_per_class:]

    n_eval_per_class = int(
        min(len(remaining_pos), len(remaining_neg)) * (eval_frac / max(1e-8, (1 - test_frac)))
    )
    eval_pos = remaining_pos[:n_eval_per_class]
    eval_neg = remaining_neg[:n_eval_per_class]
    eval_indices = np.concatenate([eval_pos, eval_neg])
    rng.shuffle(eval_indices)

    train_pos = remaining_pos[n_eval_per_class:]
    train_neg = remaining_neg[n_eval_per_class:]
    train_indices = np.concatenate([train_pos, train_neg])
    rng.shuffle(train_indices)

    return {
        "train_pool": train_indices.astype(np.int64),
        "eval": eval_indices.astype(np.int64),
        "test": test_indices.astype(np.int64),
    }


def get_split_arrays(
    activations: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice activations and labels for a given split."""
    return activations[indices], labels[indices]


def sample_train_set(
    activations: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    k: int,
    seed: int,
    balance_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample k positives plus negatives from the train pool."""
    rng = np.random.default_rng(seed)

    train_labels = labels[train_indices]
    pos_pool = train_indices[train_labels == 1]
    neg_pool = train_indices[train_labels == 0]

    if k > len(pos_pool):
        raise ValueError(f"Requested k={k} positives but only {len(pos_pool)} available")

    selected_pos = rng.choice(pos_pool, size=k, replace=False)

    if balance_mode == "balanced":
        n_neg = min(k, len(neg_pool))
        selected_neg = rng.choice(neg_pool, size=n_neg, replace=False)
    elif balance_mode == "imbalanced":
        selected_neg = neg_pool
    else:
        raise ValueError(f"Unknown balance_mode: {balance_mode}")

    selected = np.concatenate([selected_pos, selected_neg])
    rng.shuffle(selected)

    X_train = activations[selected]
    y_train = labels[selected]
    return X_train, y_train


def save_splits(splits: dict[str, np.ndarray], path: Path) -> None:
    """Save split indices to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v.tolist() for k, v in splits.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def load_splits(path: Path) -> dict[str, np.ndarray]:
    """Load split indices from JSON."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: np.asarray(v, dtype=np.int64) for k, v in raw.items()}


def save_split_manifest(
    dataset_name: str,
    splits: dict[str, np.ndarray],
    labels: np.ndarray,
    path: Path,
) -> None:
    """Save useful split metadata for auditing."""
    payload = {
        "dataset": dataset_name,
        "n_total": int(len(labels)),
        "n_pos_total": int((labels == 1).sum()),
        "n_neg_total": int((labels == 0).sum()),
        "train_pool_size": int(len(splits["train_pool"])),
        "eval_size": int(len(splits["eval"])),
        "test_size": int(len(splits["test"])),
        "eval_pos": int((labels[splits["eval"]] == 1).sum()),
        "eval_neg": int((labels[splits["eval"]] == 0).sum()),
        "test_pos": int((labels[splits["test"]] == 1).sum()),
        "test_neg": int((labels[splits["test"]] == 0).sum()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
