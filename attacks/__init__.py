"""Attack utilities for robustness and evasion studies.

The module set is a lightweight, deterministic scaffold for future frontier-level
robustness experiments.  Implementations are intentionally conservative and do
not require heavyweight model-backed dependencies.
"""

from .dilution import DilutionConfig, apply_output_dilution, build_dilution_prompt
from .monitor_aware_prompting import (
    apply_monitor_aware_prompting,
    build_monitor_aware_prompt,
)
from .reasoning_compression import build_reasoning_compression_prompt
from .surrogate_evasion import (
    SurrogateEvasionConfig,
    blend_monitor_loss,
    project_logits_to_monitor_evasion,
)

__all__ = [
    "DilutionConfig",
    "SurrogateEvasionConfig",
    "apply_monitor_aware_prompting",
    "apply_output_dilution",
    "blend_monitor_loss",
    "build_dilution_prompt",
    "build_monitor_aware_prompt",
    "build_reasoning_compression_prompt",
    "project_logits_to_monitor_evasion",
]
