"""P3: LDA probe — Fisher's linear discriminant with shrinkage.

Uses automatic shrinkage estimation to handle high-dimensional settings
and very small positive-class sizes (k=1..3).
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .base import Probe


class LDAProbe(Probe):
    name = "P3_lda"

    def __init__(self):
        self._clf = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._clf = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"
        )
        self._clf.fit(X_train, y_train)

    def score(self, X_test: np.ndarray) -> np.ndarray:
        # decision_function returns signed distance to decision boundary
        return self._clf.decision_function(X_test)
