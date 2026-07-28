"""Coverage for newly added attack and causal support modules."""

from __future__ import annotations

import numpy as np
import pytest

from attacks.dilution import build_dilution_prompt, apply_output_dilution, DilutionConfig
from attacks.monitor_aware_prompting import apply_monitor_aware_prompting, build_monitor_aware_prompt
from attacks.reasoning_compression import build_reasoning_compression_prompt
from attacks.surrogate_evasion import (
    SurrogateEvasionConfig,
    blend_monitor_loss,
    project_logits_to_monitor_evasion,
)
from causal.activation_patching import patch_activation_trajectory
from causal.intervention_controls import clamp_trajectory_noise, orthogonal_projection
from causal.steering import steer_activation_trajectory


def test_dilution_and_output_attack_is_deterministic() -> None:
    prompt = "Solve: 2+2"
    config = DilutionConfig()
    first = build_dilution_prompt(prompt, config)
    second = build_dilution_prompt(prompt, config)
    assert first == second
    diluted = apply_output_dilution("done")
    assert diluted.startswith("done\n")


def test_monitor_aware_prompting_modes() -> None:
    prompt = "Explain X."
    stealth = build_monitor_aware_prompt(prompt, concealment_style="stealth")
    evasive = apply_monitor_aware_prompting(prompt, concealment_style="evasive")
    calm = build_monitor_aware_prompt(prompt, concealment_style="calm")
    assert stealth != evasive != calm
    assert prompt in stealth and prompt in evasive and prompt in calm


def test_reasoning_compression_removes_direct_signals() -> None:
    prompt = "Please think step by step and explain your reasoning."
    compressed = build_reasoning_compression_prompt(prompt)
    assert "step by step" not in compressed.lower()
    assert "final answer only" in compressed.lower()


def test_surrogate_evasion_utilities() -> None:
    cfg = SurrogateEvasionConfig(weight=2.0)
    score = project_logits_to_monitor_evasion([0.25], direction=-1, config=cfg)
    assert score < 0.25
    blended = blend_monitor_loss(1.0, 3.0, cfg)
    assert isinstance(blended, float)


def test_causal_patch_steer_and_projection_smoke() -> None:
    traj = np.zeros((2, 3, 4), dtype=float)
    patch = np.ones((1, 4), dtype=float)
    patched = patch_activation_trajectory(traj, patch, start=1, end=2)
    assert patched.shape == traj.shape

    steered = steer_activation_trajectory(patched, np.ones(4), alpha=0.2)
    assert np.allclose(steered[:, 1, :], 1.2)

    proj = orthogonal_projection(np.arange(3.0), np.eye(3))
    assert np.allclose(proj, np.zeros(3))

    clamped = clamp_trajectory_noise(np.array([[3.0, 4.0]]), radius=1.0)
    assert np.all(np.linalg.norm(clamped, axis=-1) <= 1.0 + 1e-9)


def test_projection_handles_nonorthogonal_and_rank_deficient_bases() -> None:
    values = np.asarray([2.0, 3.0, 4.0])
    nuisance_basis = np.asarray(
        [
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    residual = orthogonal_projection(values, nuisance_basis)
    assert np.allclose(nuisance_basis.T @ residual, 0.0, atol=1e-10)
    assert residual[2] == pytest.approx(4.0)


def test_activation_patching_rejects_out_of_bounds_endpoint() -> None:
    trajectory = np.zeros((2, 4))
    patch = np.zeros((1, 4))
    with pytest.raises(ValueError, match="invalid patch interval"):
        patch_activation_trajectory(trajectory, patch, start=1, end=3)


def test_invalid_direction_rejected() -> None:
    with pytest.raises(ValueError, match="direction must be either"):
        project_logits_to_monitor_evasion(0.5, direction=3)
