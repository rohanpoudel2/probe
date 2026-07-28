"""Causal intervention helpers for activation-trajectory experiments."""

from .activation_patching import patch_activation_trajectory
from .intervention_controls import clamp_trajectory_noise, orthogonal_projection
from .steering import steer_activation_trajectory

__all__ = [
    "clamp_trajectory_noise",
    "orthogonal_projection",
    "patch_activation_trajectory",
    "steer_activation_trajectory",
]
