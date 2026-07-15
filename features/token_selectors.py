from __future__ import annotations

from typing import Dict, Tuple

Span = Tuple[int, int]


def build_single_token_span(index: int) -> Span:
    return (index, index + 1)


def suffix_span(length: int, suffix_tokens: int) -> Span:
    start = max(0, length - suffix_tokens)
    return (start, length)


def require_named_span(spans: Dict[str, Span], name: str) -> Span:
    if name not in spans:
        raise KeyError(
            f"Missing span '{name}'. Available spans: {sorted(spans.keys())}"
        )
    return spans[name]
