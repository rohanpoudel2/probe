"""P1: Logistic Regression probe — L2-regularised linear classifier."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import Probe


class LogisticProbe(Probe):
    name = "P1_logistic"
    scores_are_probabilities = True

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self._clf = None
        self._scaler = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.validate_training_data(X_train, y_train)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)
        self._clf = LogisticRegression(
            l1_ratio=0, C=self.C, solver="lbfgs", max_iter=self.max_iter
        )
        self._clf.fit(X_scaled, y_train)

    def score(self, X_test: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(self._scaler.transform(X_test))[:, 1]
