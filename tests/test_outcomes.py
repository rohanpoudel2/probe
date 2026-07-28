from data.outcomes import (
    MODEL_OUTCOME_CLASSES,
    OUTCOME_CLASS_MISSING_FINAL_ANSWER,
    OUTCOME_CLASS_MISSING_RULE_METADATA,
)


def test_missing_outcome_classes_not_included_in_model_set() -> None:
    assert OUTCOME_CLASS_MISSING_FINAL_ANSWER not in MODEL_OUTCOME_CLASSES
    assert OUTCOME_CLASS_MISSING_RULE_METADATA not in MODEL_OUTCOME_CLASSES
