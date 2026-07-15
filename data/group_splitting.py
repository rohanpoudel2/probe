from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

import numpy as np

from data.schema import TaskExample


SplitDict = Dict[str, List[TaskExample]]
PROTOCOL_SPLITS = ("train", "calibration", "eval", "test")


def _group_examples(
    examples: List[TaskExample], group_key: str
) -> Dict[str, List[TaskExample]]:
    groups: Dict[str, List[TaskExample]] = defaultdict(list)
    for ex in examples:
        group = (
            getattr(ex, group_key, None) or ex.metadata.get(group_key) or ex.example_id
        )
        groups[str(group)].append(ex)
    return groups


def _allocate_grouped_splits(
    examples: List[TaskExample],
    group_key: str,
    fractions: Dict[str, float],
    seed: int,
) -> SplitDict:
    if not examples:
        return {name: [] for name in fractions}
    if any(value < 0.0 or value >= 1.0 for value in fractions.values()):
        raise ValueError("Every split fraction must be in [0, 1)")
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(
            f"Split fractions must sum to 1, got {sum(fractions.values()):.6f}"
        )

    groups = _group_examples(examples, group_key)
    group_items = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)
    n_groups = len(group_items)

    positive_splits = [name for name, fraction in fractions.items() if fraction > 0.0]
    if n_groups < len(positive_splits):
        raise ValueError(
            f"Need at least {len(positive_splits)} groups for non-empty {positive_splits}; found {n_groups}"
        )

    raw_counts = {name: fractions[name] * n_groups for name in fractions}
    counts = {name: int(raw_counts[name]) for name in fractions}
    for name in positive_splits:
        counts[name] = max(1, counts[name])

    while sum(counts.values()) > n_groups:
        candidates = [name for name in positive_splits if counts[name] > 1]
        if not candidates:
            raise ValueError(
                "Could not allocate at least one group to every requested split"
            )
        name = max(candidates, key=lambda key: counts[key] - raw_counts[key])
        counts[name] -= 1
    while sum(counts.values()) < n_groups:
        name = max(fractions, key=lambda key: raw_counts[key] - counts[key])
        counts[name] += 1

    splits: SplitDict = {name: [] for name in fractions}
    cursor = 0
    for name in fractions:
        for _, items in group_items[cursor : cursor + counts[name]]:
            splits[name].extend(items)
        cursor += counts[name]
    return splits


def grouped_train_eval_test_split(
    examples: List[TaskExample],
    group_key: str = "question_id",
    train_frac: float = 0.7,
    eval_frac: float = 0.1,
    seed: int = 42,
) -> SplitDict:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0 <= eval_frac < 1:
        raise ValueError("eval_frac must be between 0 and 1")
    if train_frac + eval_frac >= 1:
        raise ValueError("train_frac + eval_frac must be < 1")

    return _allocate_grouped_splits(
        examples,
        group_key,
        {
            "train": train_frac,
            "eval": eval_frac,
            "test": 1.0 - train_frac - eval_frac,
        },
        seed,
    )


def grouped_train_calibration_eval_test_split(
    examples: List[TaskExample],
    group_key: str = "question_id",
    train_frac: float = 0.7,
    calibration_frac: float = 0.1,
    eval_frac: float = 0.1,
    seed: int = 42,
) -> SplitDict:
    """Create four disjoint group-level splits for frozen-threshold studies."""

    test_frac = 1.0 - train_frac - calibration_frac - eval_frac
    if (
        train_frac <= 0.0
        or calibration_frac <= 0.0
        or eval_frac <= 0.0
        or test_frac <= 0.0
    ):
        raise ValueError(
            "train, calibration, eval, and test fractions must all be positive and sum to 1"
        )
    return _allocate_grouped_splits(
        examples,
        group_key,
        {
            "train": train_frac,
            "calibration": calibration_frac,
            "eval": eval_frac,
            "test": test_frac,
        },
        seed,
    )


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
