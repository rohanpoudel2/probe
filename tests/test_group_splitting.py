from data.group_splitting import declared_protocol_split
from data.schema import TaskExample


def test_declared_protocol_splits_cannot_leak_a_group() -> None:
    examples = [
        TaskExample(
            example_id="a",
            task_family="test",
            prompt="prompt",
            label=0,
            question_id="q",
            metadata={"protocol_split": "train"},
        ),
        TaskExample(
            example_id="b",
            task_family="test",
            prompt="prompt",
            label=1,
            question_id="q",
            metadata={"protocol_split": "test"},
        ),
    ]
    import pytest

    with pytest.raises(ValueError, match="spans protocol splits"):
        declared_protocol_split(examples)
