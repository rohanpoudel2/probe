from data.group_splitting import declared_protocol_split, grouped_train_calibration_eval_test_split
from data.schema import TaskExample


def test_four_way_split_is_group_disjoint_and_complete() -> None:
    examples = [
        TaskExample(
            example_id=f"q{group}_{label}",
            task_family="test",
            prompt="prompt",
            label=label,
            question_id=f"q{group}",
        )
        for group in range(20)
        for label in (0, 1)
    ]
    splits = grouped_train_calibration_eval_test_split(examples, seed=11)
    assert set(splits) == {"train", "calibration", "eval", "test"}
    seen: set[str] = set()
    for rows in splits.values():
        groups = {row.question_id for row in rows}
        assert groups
        assert seen.isdisjoint(groups)
        seen.update(groups)
    assert seen == {f"q{group}" for group in range(20)}
    assert sum(len(rows) for rows in splits.values()) == len(examples)


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
