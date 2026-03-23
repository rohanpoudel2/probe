"""P4: Cosine Similarity probe — difference of cosine similarities to class centroids."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import Probe


class CosineProbe(Probe):
    name = "P4_cosine"

    def __init__(self):
        self._centroid_pos = None
        self._centroid_neg = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._centroid_pos = X_train[y_train == 1].mean(axis=0, keepdims=True)
        self._centroid_neg = X_train[y_train == 0].mean(axis=0, keepdims=True)

    def score(self, X_test: np.ndarray) -> np.ndarray:
        sim_pos = cosine_similarity(X_test, self._centroid_pos).ravel()
        sim_neg = cosine_similarity(X_test, self._centroid_neg).ravel()
        return sim_pos - sim_neg
