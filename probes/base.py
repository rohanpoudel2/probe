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
    requires_modified_activations: str | None = None  # "prompted", "followup", or None
    cache_suffix: str = ""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the probe on training activations and labels."""
        ...

    @abstractmethod
    def score(self, X_test: np.ndarray) -> np.ndarray:
        """Return a score for each test sample. Higher = more likely positive."""
        ...
