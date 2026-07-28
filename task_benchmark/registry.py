from __future__ import annotations

from probes import (
    CosineProbe,
    LDAProbe,
    LogisticProbe,
    CITMProbe,
    MahalanobisProbe,
    MassMeanProbe,
)

TASK_PROBE_REGISTRY = {
    "P1_logistic": LogisticProbe,
    "P2_mass_mean": MassMeanProbe,
    "P3_lda": LDAProbe,
    "P4_cosine": CosineProbe,
    "P7_mahalanobis": MahalanobisProbe,
    "P8_citm": CITMProbe,
}
