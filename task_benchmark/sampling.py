from __future__ import annotations

import numpy as np


def sample_few_shot_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    seed: int,
    balance_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    if k > len(pos_idx):
        raise ValueError(f"Requested k={k} positives but only {len(pos_idx)} available")

    selected_pos = rng.choice(pos_idx, size=k, replace=False)

    if balance_mode == "balanced":
        n_neg = min(k, len(neg_idx))
        selected_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    elif balance_mode == "imbalanced":
        selected_neg = neg_idx
    else:
        raise ValueError(f"Unknown balance_mode: {balance_mode}")

    selected = np.concatenate([selected_pos, selected_neg])
    rng.shuffle(selected)
    return X_train[selected], y_train[selected]
