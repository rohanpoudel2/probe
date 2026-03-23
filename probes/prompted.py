"""P6: Prompted Probing — logistic regression on prompted activations.

The probe itself is identical to P1 (logistic regression). The difference
is that activations come from a forward pass where a task description was
prepended to the input. From Tillman and Mossing 2025.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import Probe


class PromptedProbe(Probe):
    name = "P6_prompted"
    requires_modified_activations = "prompted"
    cache_suffix = "_prompted"

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self._clf = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._clf = LogisticRegression(
            penalty="l2", C=self.C, solver="lbfgs", max_iter=self.max_iter
        )
        self._clf.fit(X_train, y_train)

    def score(self, X_test: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X_test)[:, 1]
