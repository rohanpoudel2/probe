"""P8: Sequence-aware trajectory probe for early warning and trajectory features."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import Probe


class CITMProbe(Probe):
    """Lightweight trajectory probe using stacked trajectory-prefix features.

    The extractor stores trajectory views as cumulative concatenations, so each
    stacked prefix feature can be interpreted as a sequence of equally-sized step
    blocks. This probe reshapes the input to ``(n, trajectory_steps, step_dim)``,
    derives temporal difference features, and fits a regularised logistic probe.
    """

    name = "P8_citm"
    scores_are_probabilities = True

    def __init__(
        self,
        trajectory_steps: int | None = None,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        self.trajectory_steps = trajectory_steps
        self.C = C
        self.max_iter = max_iter
        self._clf = None
        self._scaler = None

    def _build_sequence_features(self, X_train: np.ndarray) -> np.ndarray:
        if self.trajectory_steps is None:
            raise ValueError("trajectory_steps is required for P8_citm")
        if self.trajectory_steps < 1:
            raise ValueError(
                "trajectory_steps must be positive for P8_citm sequence representation"
            )
        X = np.asarray(X_train, dtype=float)
        if X.ndim != 2:
            raise ValueError("P8_citm expects a 2D feature matrix")
        n_samples, total_dim = X.shape
        if total_dim % self.trajectory_steps != 0:
            raise ValueError(
                "P8_citm expects hidden dimension to be divisible by trajectory_steps"
            )
        step_dim = total_dim // self.trajectory_steps
        sequence = X.reshape(n_samples, self.trajectory_steps, step_dim)
        prefix_start = sequence[:, 0, :]
        prefix_end = sequence[:, -1, :]
        mean_prefix = sequence.mean(axis=1)
        if self.trajectory_steps > 1:
            deltas = np.diff(sequence, axis=1)
            mean_delta = deltas.mean(axis=1)
            delta_norm = np.linalg.norm(deltas, axis=2)
            max_delta_norm = delta_norm.max(axis=1, keepdims=True)
            last_delta = deltas[:, -1, :]
        else:
            mean_delta = np.zeros((n_samples, step_dim), dtype=float)
            max_delta_norm = np.zeros((n_samples, 1), dtype=float)
            last_delta = np.zeros((n_samples, step_dim), dtype=float)
        return np.concatenate(
            [prefix_start, prefix_end, mean_prefix, mean_delta, last_delta, max_delta_norm],
            axis=1,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.validate_training_data(X_train, y_train)
        transformed = self._build_sequence_features(X_train)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(transformed)
        self._clf = LogisticRegression(
            solver="lbfgs",
            C=self.C,
            max_iter=self.max_iter,
        )
        self._clf.fit(X_scaled, y_train)

    def score(self, X_test: np.ndarray) -> np.ndarray:
        if self._clf is None or self._scaler is None:
            raise RuntimeError("P8_citm.fit must be called before score")
        transformed = self._build_sequence_features(X_test)
        X_scaled = self._scaler.transform(transformed)
        return self._clf.predict_proba(X_scaled)[:, 1]
