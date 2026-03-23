"""P5: SAE Probe — sparse autoencoder features plus sparse logistic regression."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import Probe


class SAEProbe(Probe):
    name = "P5_sae"

    def __init__(
        self,
        top_k_features: int = 64,
        sae_release: str | None = None,
        sae_id: str | None = None,
        device: str = "cpu",
    ):
        self.top_k_features = top_k_features
        self.sae_release = sae_release
        self.sae_id = sae_id
        self.device = device
        self.sae = None
        self._feature_indices = None
        self._clf = None

    def load_sae(self):
        from sae_lens import SAE

        if not self.sae_release or not self.sae_id:
            raise RuntimeError("SAE release and id must be configured before using P5_sae")

        self.sae = SAE.from_pretrained(self.sae_release, self.sae_id)[0].to(self.device)
        self.sae.eval()

    def _encode(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.sae is None:
            self.load_sae()

        encoder_weight = getattr(self.sae, "W_enc", None)
        if encoder_weight is not None and int(encoder_weight.shape[0]) != int(X.shape[1]):
            raise ValueError(
                f"SAE input dimension {int(encoder_weight.shape[0])} does not match "
                f"activation dimension {int(X.shape[1])}"
            )

        outs = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                batch = torch.tensor(X[start : start + 256], dtype=torch.float32, device=self.device)
                feats = self.sae.encode(batch).detach().cpu().numpy()
                outs.append(feats)
        return np.concatenate(outs, axis=0)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        features = self._encode(X_train)

        pos_mean = features[y_train == 1].mean(axis=0)
        neg_mean = features[y_train == 0].mean(axis=0)
        diff = np.abs(pos_mean - neg_mean)
        self._feature_indices = np.argsort(diff)[-self.top_k_features :]

        X_selected = features[:, self._feature_indices]
        self._clf = LogisticRegression(
            penalty="l1", solver="saga", max_iter=1000, C=1.0
        )
        self._clf.fit(X_selected, y_train)

    def score(self, X_test: np.ndarray) -> np.ndarray:
        features = self._encode(X_test)
        X_selected = features[:, self._feature_indices]
        return self._clf.predict_proba(X_selected)[:, 1]
