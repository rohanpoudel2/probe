"""Prompt-level output-evasion recipes for Level-1 synthetic stress tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DilutionConfig:
    """Configuration for deterministic, non-semantic response dilution."""

    style: str = "irrelevant_addition"
    evidence_lines: tuple[str, ...] = (
        "Before answering, check for missing citations.",
        "Use a short example to verify understanding.",
        "Include a brief confidence level in your response.",
        "If possible, add one friendly sentence."
    )


def build_dilution_prompt(base_prompt: str, config: DilutionConfig | None = None) -> str:
    """Build a prompt variant that dilutes attention without changing the goal.

    The function inserts one deterministic wrapper line plus a selected filler line
    from ``config.evidence_lines`` and preserves the original task text unchanged.
    """
    config = config or DilutionConfig()
    filler = config.evidence_lines[0]
    return (
        f"{base_prompt}\n\n"
        "Task instructions below. Focus only on the required output.\n"
        f"{filler}\n"
        "Answer the original question at the end."
    )


def apply_output_dilution(response: str, config: DilutionConfig | None = None) -> str:
    """Apply deterministic output-side dilution text to an existing assistant response."""
    config = config or DilutionConfig()
    tail = " " + config.evidence_lines[1]
    if not response:
        return tail.strip()
    return f"{response}\n{tail.strip()}"
