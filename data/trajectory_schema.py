from __future__ import annotations

import re
from math import ceil

TRAJECTORY_PREFIX_VIEW_RE = re.compile(r"^trajectory_prefix_p(\d+)$")
TRAJECTORY_PREFIX_STACK_VIEW_RE = re.compile(r"^trajectory_prefix_stack_p(\d+)$")
TRAJECTORY_PROMPT_END_VIEW = "trajectory_prompt_end"
TRAJECTORY_BASIS = "assistant_response_tokens_v1"


def trajectory_prefix_view(percentile: int) -> str:
    p = int(percentile)
    if not 1 <= p <= 100:
        raise ValueError("trajectory percentiles must be in [1, 100]")
    return f"trajectory_prefix_p{p}"


def parse_trajectory_prefix_view(view_name: str) -> int | None:
    match = TRAJECTORY_PREFIX_VIEW_RE.match(view_name)
    if not match:
        return None
    return int(match.group(1))


def trajectory_prefix_stack_view(percentile: int) -> str:
    p = int(percentile)
    if not 1 <= p <= 100:
        raise ValueError("trajectory percentiles must be in [1, 100]")
    return f"trajectory_prefix_stack_p{p}"


def parse_trajectory_prefix_stack_view(view_name: str) -> int | None:
    match = TRAJECTORY_PREFIX_STACK_VIEW_RE.match(view_name)
    if not match:
        return None
    return int(match.group(1))


def normalize_percentiles(percentiles: list[int] | tuple[int, ...] | None) -> list[int]:
    if not percentiles:
        return []
    normalized = sorted({int(value) for value in percentiles})
    for value in normalized:
        if not 1 <= int(value) <= 100:
            raise ValueError("trajectory_prefix_percentiles must be in [1, 100]")
    return normalized


def build_trajectory_prefix_views(
    percentiles: list[int] | tuple[int, ...] | None,
    token_count: int,
) -> list[tuple[str, int]]:
    normalized = normalize_percentiles(percentiles)
    if not normalized or token_count < 1:
        return []
    windows: list[tuple[str, int]] = []
    for percentile in normalized:
        window = max(1, min(token_count, ceil(token_count * (percentile / 100))))
        windows.append((trajectory_prefix_view(percentile), window))
    return windows
