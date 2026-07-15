import numpy as np
import pytest

from data.control_features import transform_bundle


def _write_bundle(path):
    np.savez(
        path,
        answer=np.arange(12, dtype=float).reshape(6, 2),
        labels=np.asarray([0, 1, 0, 1, 0, 1]),
        example_ids=np.asarray([f"e{i}" for i in range(6)]),
        question_ids=np.asarray([f"q{i}" for i in range(6)]),
        model_name=np.asarray("fixture"),
    )


def test_permute_features_breaks_alignment_but_preserves_labels_and_ids(tmp_path) -> None:
    source = tmp_path / "source.npz"
    target = tmp_path / "target.npz"
    _write_bundle(source)
    transform_bundle(source, target, "permute_features", seed=3)
    original = np.load(source)
    transformed = np.load(target)
    assert np.array_equal(transformed["labels"], original["labels"])
    assert np.array_equal(transformed["example_ids"], original["example_ids"])
    assert not np.array_equal(transformed["answer"], original["answer"])
    assert transformed["model_name"].item() == "fixture"


def test_legacy_shuffle_rows_is_rejected(tmp_path) -> None:
    source = tmp_path / "source.npz"
    target = tmp_path / "target.npz"
    _write_bundle(source)
    with pytest.raises(ValueError, match="preserves the task"):
        transform_bundle(source, target, "shuffle_rows")
