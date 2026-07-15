"""Abstract base class for all activation probes."""

from abc import ABC, abstractmethod
import numpy as np


class Probe(ABC):
    """Base interface for activation probes.

    All probes implement fit() and score(). The sweep orchestrator calls
    them interchangeably — the main behavioural difference is which activation
    cache each probe reads from.
    """

    name: str = "base"
    requires_modified_activations: str | None = None  # "prompted" or None
    cache_suffix: str = ""
    scores_are_probabilities: bool = False
    minimum_class_counts: dict[int, int] = {0: 1, 1: 1}

    def validate_training_data(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X = np.asarray(X_train)
        y = np.asarray(y_train)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("Probe training features and labels must be aligned 2D/1D arrays")
        if not np.all(np.isfinite(X)):
            raise ValueError("Probe training features must be finite")
        for label, minimum in self.minimum_class_counts.items():
            observed = int(np.sum(y == label))
            if observed < minimum:
                raise ValueError(
                    f"{self.name} requires at least {minimum} label-{label} examples; found {observed}"
                )

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the probe on training activations and labels."""
        ...

    @abstractmethod
    def score(self, X_test: np.ndarray) -> np.ndarray:
        """Return a score for each test sample. Higher = more likely positive."""
        ...
