from __future__ import annotations

from typing import Dict, List

from data.schema import TaskExample


SplitDict = Dict[str, List[TaskExample]]
PROTOCOL_SPLITS = ("train", "calibration", "eval", "test")


def declared_protocol_split(
    examples: List[TaskExample],
    group_key: str = "question_id",
) -> SplitDict:
    """Use immutable split assignments carried by labeled rollout records."""

    splits: SplitDict = {name: [] for name in PROTOCOL_SPLITS}
    group_assignments: dict[str, str] = {}
    for example in examples:
        split = example.metadata.get("protocol_split")
        if split not in PROTOCOL_SPLITS:
            raise ValueError(
                f"Example {example.example_id} is missing a valid declared protocol_split"
            )
        group = str(
            getattr(example, group_key, None)
            or example.metadata.get(group_key)
            or example.example_id
        )
        previous = group_assignments.setdefault(group, split)
        if previous != split:
            raise ValueError(
                f"Group {group} spans protocol splits {previous} and {split}"
            )
        splits[split].append(example)
    return splits
