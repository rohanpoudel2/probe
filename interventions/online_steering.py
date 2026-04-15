from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass
class OnlineSteeringStats:
    n_forward_calls: int = 0
    n_applied: int = 0
    score_sum: float = 0.0

    @property
    def mean_score(self) -> float:
        if self.n_forward_calls == 0:
            return float("nan")
        return self.score_sum / self.n_forward_calls


def resolve_transformer_layers(model) -> Iterable[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise ValueError("Unsupported model architecture for layer hooks")


def _project_scores(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return hidden @ direction


@contextmanager
def steering_hooks(
    model,
    layer_indices: list[int],
    direction: np.ndarray,
    alpha: float,
    threshold: float | None = None,
):
    layers = list(resolve_transformer_layers(model))
    if not layer_indices:
        raise ValueError("layer_indices must be non-empty")
    if max(layer_indices) >= len(layers) or min(layer_indices) < 0:
        raise IndexError(f"Requested hook layer outside model range: {layer_indices}")

    device = next(model.parameters()).device
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    norm = torch.linalg.norm(direction_t)
    if float(norm) > 0.0:
        direction_t = direction_t / norm

    stats = OnlineSteeringStats()
    handles = []

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[-1] != direction_t.shape[0]:
            return output

        current = hidden[:, -1, :]
        scores = _project_scores(current.float(), direction_t)
        stats.n_forward_calls += int(scores.numel())
        stats.score_sum += float(scores.sum().item())

        if threshold is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = scores > threshold
        if not torch.any(mask):
            return output

        steered = hidden.clone()
        steered[mask, -1, :] = steered[mask, -1, :] - alpha * direction_t
        stats.n_applied += int(mask.sum().item())
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    for idx in layer_indices:
        handles.append(layers[idx].register_forward_hook(_hook))

    try:
        yield stats
    finally:
        for handle in handles:
            handle.remove()
