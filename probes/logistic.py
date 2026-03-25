"""P1: Logistic Regression probe — L2-regularised linear classifier."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import Probe


class LogisticProbe(Probe):
    name = "P1_logistic"

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self._clf = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._clf = LogisticRegression(
            l1_ratio=0, C=self.C, solver="lbfgs", max_iter=self.max_iter
        )
        self._clf.fit(X_train, y_train)

    def score(self, X_test: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X_test)[:, 1]
