from .base import Probe
from .logistic import LogisticProbe
from .mass_mean import MassMeanProbe
from .lda import LDAProbe
from .cosine import CosineProbe
from .prompted import PromptedProbe
from .mahalanobis import MahalanobisProbe
from .followup import FollowupProbe

PROBE_REGISTRY = {
    "P1_logistic": LogisticProbe,
    "P2_mass_mean": MassMeanProbe,
    "P3_lda": LDAProbe,
    "P4_cosine": CosineProbe,
    "P6_prompted": PromptedProbe,
    "P7_mahalanobis": MahalanobisProbe,
    "P8_followup": FollowupProbe,
}
