from .base import Probe
from .logistic import LogisticProbe
from .mass_mean import MassMeanProbe
from .lda import LDAProbe
from .cosine import CosineProbe
from .mahalanobis import MahalanobisProbe

__all__ = [
    "CosineProbe",
    "LDAProbe",
    "LogisticProbe",
    "MahalanobisProbe",
    "MassMeanProbe",
    "Probe",
    "PROBE_REGISTRY",
]

PROBE_REGISTRY = {
    "P1_logistic": LogisticProbe,
    "P2_mass_mean": MassMeanProbe,
    "P3_lda": LDAProbe,
    "P4_cosine": CosineProbe,
    "P7_mahalanobis": MahalanobisProbe,
}
