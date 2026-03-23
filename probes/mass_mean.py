"""P2: Mass-Mean Probe — projection onto class mean difference vector.

No optimisation required. Score is the dot product of each sample with
the direction from the negative centroid to the positive centroid.
From Marks and Tegmark 2024.
"""

import numpy as np
from .base import Probe


class MassMeanProbe(Probe):
    name = "P2_mass_mean"

    def __init__(self):
        self._direction = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        mu_pos = X_train[y_train == 1].mean(axis=0)
        mu_neg = X_train[y_train == 0].mean(axis=0)
        self._direction = mu_pos - mu_neg
        norm = np.linalg.norm(self._direction)
        if norm > 0:
            self._direction /= norm

    def score(self, X_test: np.ndarray) -> np.ndarray:
        return X_test @ self._direction
