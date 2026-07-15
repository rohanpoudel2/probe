"""P6: Prompted Probing — logistic regression on prompted activations.

The probe itself is identical to P1 (logistic regression). The difference
is that activations come from a forward pass where a task description was
prepended to the input. From Tillman and Mossing 2025.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import Probe


class PromptedProbe(Probe):
    name = "P6_prompted"
    requires_modified_activations = "prompted"
    cache_suffix = "_prompted"
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
