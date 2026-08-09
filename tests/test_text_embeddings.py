import numpy as np
import pytest
import torch

from cli.extract_text_embeddings import (
    _validate_resumable_cache,
    pool_hidden_states,
    render_embedding_input,
)
from cli.run_embedding_baselines import _assert_compatible
from data.schema import TaskExample
from data.text_embedding_cache import (
    TEXT_EMBEDDING_SCHEMA_VERSION,
    atomic_save_text_embedding_cache,
    load_text_embedding_cache,
)
from data.text_views import (
    is_valid_text_view,
    response_prefix_text_view,
    text_view,
)


def test_last_token_pool_handles_left_and_right_padding() -> None:
    hidden = torch.tensor(
        [
            [[0.0], [1.0], [2.0]],
            [[3.0], [4.0], [5.0]],
        ]
    )
    right_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    left_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    assert pool_hidden_states(hidden, right_mask, "last").squeeze(-1).tolist() == [1.0, 5.0]
    assert pool_hidden_states(hidden, left_mask, "last").squeeze(-1).tolist() == [2.0, 5.0]


def test_mean_pool_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert pool_hidden_states(hidden, mask, "mean").item() == 2.0


def _metadata(**overrides):
    base = {
        "schema_version": TEXT_EMBEDDING_SCHEMA_VERSION,
        "task_name": "sycophancy",
        "view": "answer_text",
        "dataset_sha256": "1" * 64,
        "code_revision": "2" * 40,
        "code_dirty": False,
        "embedding_model_id": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_model_revision": "3" * 40,
        "embedding_tokenizer_revision": "3" * 40,
        "embedding_spec_sha256": "4" * 64,
        "embedding_config_sha256": "5" * 64,
        "pooling": "last",
        "padding_side": "left",
        "normalized": True,
        "max_length": 8192,
        "instruction": "Represent this interaction.",
        "instruction_format": "Instruct: {instruction}\nQuery:{text}",
        "requested_batch_size": 16,
        "effective_batch_size": 16,
        "resolved_device": "cpu",
        "model_parameter_dtype": "torch.float32",
        "monitored_model_id": "org/model",
        "monitored_model_revision": "6" * 40,
        "monitored_tokenizer_revision": "6" * 40,
    }
    base.update(overrides)
    return base


def _arrays():
    return {
        "embeddings": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "labels": np.asarray([0, 1], dtype=np.int64),
        "example_ids": np.asarray(["e0", "e1"]),
        "question_ids": np.asarray(["q0", "q0"]),
        "protocol_splits": np.asarray(["train", "train"]),
        "text_sha256": np.asarray(["7" * 64, "8" * 64]),
        "normalized_text_sha256": np.asarray(["b" * 64, "c" * 64]),
        "embedding_input_sha256": np.asarray(["9" * 64, "a" * 64]),
        "original_token_lengths": np.asarray([10, 11], dtype=np.int64),
        "truncated": np.asarray([False, False]),
    }


def test_embedding_cache_round_trip_and_compatibility(tmp_path) -> None:
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    atomic_save_text_embedding_cache(first_path, arrays=_arrays(), metadata=_metadata())
    atomic_save_text_embedding_cache(second_path, arrays=_arrays(), metadata=_metadata())
    first = load_text_embedding_cache(first_path)
    second = load_text_embedding_cache(second_path)
    assert first["embeddings"].shape == (2, 2)
    _assert_compatible(first, second)

    incompatible_path = tmp_path / "incompatible.npz"
    atomic_save_text_embedding_cache(
        incompatible_path,
        arrays=_arrays(),
        metadata=_metadata(instruction="A different registered instruction."),
    )
    incompatible = load_text_embedding_cache(incompatible_path)
    with pytest.raises(ValueError, match="instruction differs"):
        _assert_compatible(first, incompatible)


def test_resume_requires_matching_embedding_cache_identity(tmp_path) -> None:
    metadata = _metadata()
    _validate_resumable_cache(
        tmp_path / "cache.npz",
        metadata,
        {"dataset_sha256": metadata["dataset_sha256"], "view": "answer_text"},
    )
    with pytest.raises(ValueError, match="dataset_sha256"):
        _validate_resumable_cache(
            tmp_path / "cache.npz",
            metadata,
            {"dataset_sha256": "0" * 64},
        )


def test_embedding_input_uses_locked_instruction_format() -> None:
    spec = {
        "instruction": "Represent this interaction.",
        "instruction_format": "Instruct: {instruction}\nQuery:{text}",
    }
    assert render_embedding_input("hello", spec) == (
        "Instruct: Represent this interaction.\nQuery:hello"
    )


def test_prompt_cache_rejects_duplicate_normalized_prompts_across_groups(tmp_path) -> None:
    arrays = _arrays()
    arrays["question_ids"] = np.asarray(["q0", "q1"])
    arrays["normalized_text_sha256"] = np.asarray(["b" * 64, "b" * 64])
    path = tmp_path / "duplicate_prompts.npz"
    atomic_save_text_embedding_cache(
        path,
        arrays=arrays,
        metadata=_metadata(view="prompt_text"),
    )
    with pytest.raises(ValueError, match="repeats normalized prompts"):
        load_text_embedding_cache(path)


def test_response_prefix_text_views_are_deterministic_and_nonempty() -> None:
    example = TaskExample(
        example_id="e",
        task_family="test",
        prompt="prompt",
        label=0,
        assistant_response="abcdefghij",
    )
    view = response_prefix_text_view(25)
    assert is_valid_text_view(view)
    assert text_view(example, view) == "abc"
