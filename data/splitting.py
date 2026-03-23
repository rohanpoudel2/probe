"""Reproducible dataset splitting and few-shot sampling."""

import json
from pathlib import Path

import numpy as np
from datasets import Dataset


def make_splits(
    dataset: Dataset, seed: int = 42, test_frac: float = 0.15, eval_frac: float = 0.15
) -> dict:
    """Split dataset into train_pool / eval / test with balanced test set.

    Args:
        dataset: Dataset with 'text' and 'label' columns.
        seed: Random seed for reproducibility.
        test_frac: Fraction of data for test set.
        eval_frac: Fraction of data for eval set.

    Returns:
        Dict with 'train_pool', 'eval', 'test' keys mapping to index arrays.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset["label"])
    n = len(labels)

    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    # Balanced test set: equal pos and neg
    n_test_per_class = int(min(len(pos_indices), len(neg_indices)) * test_frac)
    test_pos = pos_indices[:n_test_per_class]
    test_neg = neg_indices[:n_test_per_class]
    test_indices = np.concatenate([test_pos, test_neg])

    # Remaining after test
    remaining_pos = pos_indices[n_test_per_class:]
    remaining_neg = neg_indices[n_test_per_class:]

    # Eval set (also balanced)
    n_eval_per_class = int(min(len(remaining_pos), len(remaining_neg)) * (eval_frac / (1 - test_frac)))
    eval_pos = remaining_pos[:n_eval_per_class]
    eval_neg = remaining_neg[:n_eval_per_class]
    eval_indices = np.concatenate([eval_pos, eval_neg])

    # Train pool: everything else
    train_pos = remaining_pos[n_eval_per_class:]
    train_neg = remaining_neg[n_eval_per_class:]
    train_indices = np.concatenate([train_pos, train_neg])

    return {
        "train_pool": train_indices,
        "eval": eval_indices,
        "test": test_indices,
    }


def sample_train_set(
    activations: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    k: int,
    seed: int,
    balance_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a training set of k positives + negatives from the train pool.

    Args:
        activations: Full activation array, shape [N, D].
        labels: Full label array, shape [N].
        train_indices: Indices into activations/labels for the train pool.
        k: Number of positive examples to sample.
        seed: Random seed.
        balance_mode: 'balanced' (k neg) or 'imbalanced' (all available neg).

    Returns:
        (X_train, y_train) numpy arrays.
    """
    rng = np.random.default_rng(seed)

    train_labels = labels[train_indices]
    pos_pool = train_indices[train_labels == 1]
    neg_pool = train_indices[train_labels == 0]

    if k > len(pos_pool):
        raise ValueError(
            f"Requested k={k} positives but only {len(pos_pool)} available"
        )

    # Sample k positives
    selected_pos = rng.choice(pos_pool, size=k, replace=False)

    # Sample negatives
    if balance_mode == "balanced":
        n_neg = min(k, len(neg_pool))
        selected_neg = rng.choice(neg_pool, size=n_neg, replace=False)
    elif balance_mode == "imbalanced":
        selected_neg = neg_pool  # use all negatives
    else:
        raise ValueError(f"Unknown balance_mode: {balance_mode}")

    selected = np.concatenate([selected_pos, selected_neg])
    X_train = activations[selected]
    y_train = labels[selected]
    return X_train, y_train


def save_splits(splits: dict, path: Path):
    """Save split indices to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: v.tolist() for k, v in splits.items()}
    with open(path, "w") as f:
        json.dump(serialisable, f)


def load_splits(path: Path) -> dict:
    """Load split indices from JSON."""
    with open(path) as f:
        raw = json.load(f)
    return {k: np.array(v) for k, v in raw.items()}
