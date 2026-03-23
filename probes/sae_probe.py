"""P5: SAE Probe — sparse autoencoder feature selection + sparse logistic regression.

Only runs on Gemma models where Gemma Scope SAEs are available.
From Kantamneni et al. 2025.

Steps:
1. Load a pre-trained SAE for the target layer.
2. Encode activations through the SAE to get sparse feature vectors.
3. Select top-k features by mean activation difference between classes.
4. Train sparse logistic regression on the selected features.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import Probe


class SAEProbe(Probe):
    name = "P5_sae"

    def __init__(self, top_k_features: int = 64, sae=None):
        self.top_k_features = top_k_features
        self.sae = sae  # Pre-loaded SAE model (set externally)
        self._feature_indices = None
        self._clf = None

    def _encode(self, X: np.ndarray) -> np.ndarray:
        """Encode activations through SAE to get sparse features.

        If no SAE is loaded, this is a no-op (returns X unchanged).
        In practice the SAE should be set before calling fit/score.
        """
        if self.sae is None:
            return X

        import torch
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            # sae-lens SAE expects [batch, d_model], returns feature activations
            features = self.sae.encode(X_t)
            return features.numpy()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        features = self._encode(X_train)

        # Select top-k features by mean activation difference
        pos_mean = features[y_train == 1].mean(axis=0)
        neg_mean = features[y_train == 0].mean(axis=0)
        diff = np.abs(pos_mean - neg_mean)
        self._feature_indices = np.argsort(diff)[-self.top_k_features :]

        # Train sparse logistic regression on selected features
        X_selected = features[:, self._feature_indices]
        self._clf = LogisticRegression(
            penalty="l1", solver="saga", max_iter=1000, C=1.0
        )
        self._clf.fit(X_selected, y_train)

    def score(self, X_test: np.ndarray) -> np.ndarray:
        features = self._encode(X_test)
        X_selected = features[:, self._feature_indices]
        return self._clf.predict_proba(X_selected)[:, 1]
