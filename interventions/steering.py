from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SteeringConfig:
    alpha: float = 1.0
    threshold: Optional[float] = None


def should_apply(score: float, threshold: Optional[float]) -> bool:
    return threshold is None or score > threshold


def steer_activation(h: np.ndarray, direction: np.ndarray, score: float, cfg: SteeringConfig) -> np.ndarray:
    if not should_apply(score, cfg.threshold):
        return h
    return h - cfg.alpha * direction
