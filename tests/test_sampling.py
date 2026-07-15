import numpy as np
import pytest

from task_benchmark.sampling import FewShotSelection, sample_few_shot_train


def _paired_fixture(n_groups: int = 6):
    groups = np.repeat([f"q{index}" for index in range(n_groups)], 2)
    labels = np.tile([0, 1], n_groups)
    features = np.column_stack([np.arange(len(labels)), labels])
    return features, labels, groups


def test_balanced_sampling_uses_matched_groups() -> None:
    X, y, groups = _paired_fixture()
    sample = sample_few_shot_train(
        X,
        y,
        k=3,
        seed=7,
        balance_mode="balanced",
        group_ids=groups,
        return_selection=True,
    )
    assert isinstance(sample, FewShotSelection)
    assert len(sample.y) == 6
    assert int(np.sum(sample.y == 0)) == 3
    assert int(np.sum(sample.y == 1)) == 3
    selected_groups, counts = np.unique(sample.group_ids, return_counts=True)
    assert len(selected_groups) == 3
    assert counts.tolist() == [2, 2, 2]
    for group in selected_groups:
        assert set(sample.y[sample.group_ids == group]) == {0, 1}


def test_balanced_sampling_refuses_unmatched_rows() -> None:
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.asarray([1, 1, 1, 0, 0, 0])
    groups = np.asarray(["p0", "p1", "p2", "n0", "n1", "n2"])
    with pytest.raises(ValueError, match="matched scenario groups"):
        sample_few_shot_train(
            X, y, k=1, seed=0, balance_mode="balanced", group_ids=groups
        )


def test_group_ids_are_mandatory() -> None:
    X, y, _ = _paired_fixture()
    with pytest.raises(TypeError):
        sample_few_shot_train(X, y, k=1, seed=0, balance_mode="balanced")
