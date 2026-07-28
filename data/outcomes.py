from __future__ import annotations

"""Shared annotation outcome constants used across labeling and reporting."""

OUTCOME_CLASS_TARGET_ALIGNED = "target_aligned"
OUTCOME_CLASS_CORRECT_NON_TARGET = "correct_non_target"
OUTCOME_CLASS_OTHER_WRONG_OR_AMBIGUOUS = "other_wrong_or_ambiguous"
OUTCOME_CLASS_MISSING_FINAL_ANSWER = "missing_final_answer"
OUTCOME_CLASS_MISSING_RULE_METADATA = "missing_rule_metadata"

# Core outcome classes kept in the final modeling dataset (label-aware examples only).
MODEL_OUTCOME_CLASSES = (
    OUTCOME_CLASS_TARGET_ALIGNED,
    OUTCOME_CLASS_CORRECT_NON_TARGET,
    OUTCOME_CLASS_OTHER_WRONG_OR_AMBIGUOUS,
)

# Outcome classes that still indicate exclusion from the study.
EXCLUDED_OUTCOME_CLASSES = (
    OUTCOME_CLASS_MISSING_FINAL_ANSWER,
    OUTCOME_CLASS_MISSING_RULE_METADATA,
)

ALL_OUTCOME_CLASSES = MODEL_OUTCOME_CLASSES + EXCLUDED_OUTCOME_CLASSES


def is_valid_model_outcome(value: str) -> bool:
    return value in MODEL_OUTCOME_CLASSES


def is_excluded_outcome(value: str) -> bool:
    return value in EXCLUDED_OUTCOME_CLASSES


def is_valid_outcome(value: str) -> bool:
    return value in ALL_OUTCOME_CLASSES
