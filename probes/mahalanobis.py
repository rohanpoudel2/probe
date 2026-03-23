"""P7: Mahalanobis Distance probe — anomaly scoring from negative-class distribution.

Fits only on the negative class: computes mean and covariance of normal
(non-spam) activations, then scores test samples by their Mahalanobis
distance from that distribution. Higher distance = more anomalous = more
likely positive. Uses LedoitWolf shrinkage for stable covariance estimation.
From MAAD benchmark 2024.
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import LedoitWolf

from .base import Probe


class MahalanobisProbe(Probe):
    name = "P7_mahalanobis"

    def __init__(self):
        self._mean = None
        self._precision = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Fit on negative class only (the "normal" distribution)
        X_neg = X_train[y_train == 0]

        self._mean = X_neg.mean(axis=0)

        # LedoitWolf gives a well-conditioned covariance even when N < D
        lw = LedoitWolf()
        lw.fit(X_neg)
        self._precision = lw.precision_

    def score(self, X_test: np.ndarray) -> np.ndarray:
        scores = np.array([
            mahalanobis(x, self._mean, self._precision)
            for x in X_test
        ])
        return scores
