"""Prompt transformations that reduce exposed reasoning signal."""

from __future__ import annotations

import re


def build_reasoning_compression_prompt(prompt: str) -> str:
    """Transform a prompt so the model is encouraged to avoid reasoning details.

    The transformation is intentionally conservative:
      1) remove direct chain-of-thought requests,
      2) preserve the task payload and constraints,
      3) add a short no-explanation instruction.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    cleaned = re.sub(r"\b(think step by step|explain your reasoning|show your work)\b.*", "", prompt, flags=re.IGNORECASE)
    cleaned = cleaned.rstrip()
    return (
        f"{cleaned}\n\n"
        "Reply with the final answer only; do not include reasoning traces or intermediate steps."
    )
