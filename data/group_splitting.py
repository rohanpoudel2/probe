from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from data.schema import TaskExample


SplitDict = Dict[str, List[TaskExample]]


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

    groups: Dict[str, List[TaskExample]] = defaultdict(list)
    for ex in examples:
        group = getattr(ex, group_key, None) or ex.metadata.get(group_key) or ex.example_id
        groups[str(group)].append(ex)

    group_items = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)

    n_groups = len(group_items)
    n_train = max(1, int(round(train_frac * n_groups))) if n_groups > 1 else n_groups
    n_eval = int(round(eval_frac * n_groups)) if n_groups > 2 else 0
    if n_groups >= 3 and n_eval == 0:
        n_eval = 1
    if n_train + n_eval >= n_groups and n_groups >= 3:
        n_train = max(1, n_groups - n_eval - 1)

    train_groups = set(g for g, _ in group_items[:n_train])
    eval_groups = set(g for g, _ in group_items[n_train:n_train + n_eval])
    test_groups = set(g for g, _ in group_items[n_train + n_eval:])

    # If the dataset is tiny, guarantee a non-empty test split.
    if not test_groups and eval_groups:
        moved = next(iter(eval_groups))
        eval_groups.remove(moved)
        test_groups.add(moved)

    splits: SplitDict = {"train": [], "eval": [], "test": []}
    for group, items in group_items:
        if group in train_groups:
            splits["train"].extend(items)
        elif group in eval_groups:
            splits["eval"].extend(items)
        else:
            splits["test"].extend(items)
    return splits
