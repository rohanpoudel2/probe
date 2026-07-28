"""Utilities for lightweight surrogate monitor-aware evasion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurrogateEvasionConfig:
    """Configuration for deterministic surrogate-loss shaping."""

    weight: float = 1.0
    clip: float = 2.0


def project_logits_to_monitor_evasion(
    logits: list[float] | tuple[float, ...] | float,
    direction: int,
    config: SurrogateEvasionConfig | None = None,
) -> float:
    """Project a scalar monitor score away from the target class.

    ``direction`` should be +1 (increase monitor score) or -1 (decrease monitor score).
    """
    config = config or SurrogateEvasionConfig()
    score = float(logits) if isinstance(logits, (int, float)) else float(logits[0])
    direction = int(direction)
    if direction not in (-1, 1):
        raise ValueError("direction must be either +1 or -1")
    if not 0.0 <= config.weight <= 10.0:
        raise ValueError("weight must be in [0, 10]")
    return max(-config.clip, min(config.clip, score + direction * config.weight))


def blend_monitor_loss(
    monitor_loss: float,
    task_loss: float,
    config: SurrogateEvasionConfig | None = None,
) -> float:
    """Combine task preservation and monitor-evasion losses.

    ``weight`` is the registered coefficient on the monitor objective. A caller
    minimizes the returned loss and is responsible for defining ``monitor_loss``
    with the desired evasion sign.
    """
    config = config or SurrogateEvasionConfig()
    if not 0.0 <= config.weight <= 10.0:
        raise ValueError("weight must be in [0, 10]")
    return float(task_loss + config.weight * monitor_loss)
