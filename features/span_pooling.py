from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

Span = Tuple[int, int]


class SpanPoolingError(ValueError):
    pass


def mean_pool_span(hidden_states: np.ndarray, span: Span) -> np.ndarray:
    start, end = span
    if start < 0 or end > hidden_states.shape[0] or start > end:
        raise SpanPoolingError(f"Invalid span {span} for hidden state shape {hidden_states.shape}")
    if start == end:
        return np.zeros(hidden_states.shape[1], dtype=hidden_states.dtype)
    return hidden_states[start:end].mean(axis=0)


def last_token_in_span(hidden_states: np.ndarray, span: Span) -> np.ndarray:
    start, end = span
    if start < 0 or end > hidden_states.shape[0] or start > end:
        raise SpanPoolingError(f"Invalid span {span} for hidden state shape {hidden_states.shape}")
    if start == end:
        return np.zeros(hidden_states.shape[1], dtype=hidden_states.dtype)
    return hidden_states[end - 1]


def pool_named_spans(hidden_states: np.ndarray, spans: Dict[str, Span], mode: str = "mean") -> Dict[str, np.ndarray]:
    pooled: Dict[str, np.ndarray] = {}
    for name, span in spans.items():
        if mode == "mean":
            pooled[name] = mean_pool_span(hidden_states, span)
        elif mode == "last":
            pooled[name] = last_token_in_span(hidden_states, span)
        else:
            raise SpanPoolingError(f"Unsupported pooling mode: {mode}")
    return pooled
