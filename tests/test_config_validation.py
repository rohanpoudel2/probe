import pytest
import numpy as np
from pathlib import Path

from cli.common import load_yaml
from cli.validate_multimodel_config import (
    _validate_feature_directory,
    validate_config,
)


def _config():
    return {
        "protocol_version": "frontier-v1",
        "protocol_stage": "pilot",
        "results_dir": "results",
        "max_fpr": 0.01,
        "min_calibration_negatives": 1000,
        "k_values": "1,4,8",
        "seeds": 10,
        "balance_modes": "balanced",
        "run_falsification_suite": False,
        "falsification_registry": "experiments/protocol/falsification_registry.yaml",
        "task_pairs": [{"source_task": "a", "target_task": "b"}],
        "models": [
            {
                "name": "model-a",
                "model_id": "org/model-a",
                "model_revision": "abcdef1234567",
                "family": "family-a",
                "feature_dirs": {"a": "features/a", "b": "features/b"},
            }
        ],
    }


def test_pilot_config_accepts_pinned_model() -> None:
    validate_config(_config(), check_paths=False, final_protocol=False)


def test_final_config_requires_three_families_and_preregistration() -> None:
    config = _config()
    config["protocol_stage"] = "frozen"
    config["min_calibration_negatives"] = 10_000
    config["k_values"] = "1,2,4,8,16,32"
    with pytest.raises(ValueError, match="three genuinely different"):
        validate_config(config, check_paths=False, final_protocol=True)


def test_moving_model_revision_is_rejected() -> None:
    config = _config()
    config["models"][0]["model_revision"] = "main"
    with pytest.raises(ValueError, match="pinned"):
        validate_config(config, check_paths=False, final_protocol=False)


def test_pilot_validates_the_registered_embedding_lock() -> None:
    config = _config()
    config["text_embedding_model_lock"] = (
        "experiments/baselines/text_embedding_models.yaml"
    )
    config["text_embedding_model_key"] = "qwen3_embedding_0_6b"
    config["text_embedding_views"] = [
        "prompt_text",
        "answer_text",
        "transcript_text",
    ]
    validate_config(config, check_paths=False, final_protocol=False)


def test_black_box_execution_requires_complete_inputs() -> None:
    config = _config()
    config.update(
        {
            "run_black_box_baselines": True,
            "black_box_baselines": [
                "B1_text_tfidf",
                "B2_text_embedding_logistic",
                "B3_llm_judge_zero_shot",
                "B3_llm_judge_few_shot",
                "B4_output_confidence_logistic",
            ],
            "text_embedding_model_lock": "experiments/baselines/text_embedding_models.yaml",
            "text_embedding_model_key": "qwen3_embedding_0_6b",
            "text_embedding_views": [
                "prompt_text",
                "answer_text",
                "transcript_text",
            ],
            "llm_judge_model_lock": "experiments/baselines/llm_judge_models.yaml",
            "llm_judge_model_key": "phi4_14b",
            "llm_judge_views": [
                "prompt_text",
                "answer_text",
                "transcript_text",
            ],
            "llm_judge_modes": ["zero_shot", "few_shot"],
        }
    )
    with pytest.raises(ValueError, match="must define labeled_data"):
        validate_config(config, check_paths=False, final_protocol=False)


def test_legacy_protocol_manifests_are_blocked() -> None:
    for path in [
        Path("experiments/controls/_legacy/honesty_auxiliary_manifest.yaml"),
        Path("experiments/controls/_legacy/neurips_final_manifest.yaml"),
        Path("experiments/controls/neurips_final_manifest.yaml"),
    ]:
        with pytest.raises(ValueError, match="Legacy protocol is intentionally blocked"):
            validate_config(load_yaml(path), check_paths=False, final_protocol=False)


def _write_feature_bundle(
    directory: Path,
    *,
    schema_version: str,
    allow_truncation: bool = False,
    truncated: bool = False,
    filename: str = "train_layer0.npz",
    question_ids: tuple[str, str] = ("g0", "g1"),
) -> None:
    original_counts = np.asarray([5, 4], dtype=np.int64)
    retained_counts = np.asarray([4 if truncated else 5, 4], dtype=np.int64)
    arrays = {
        "answer": np.zeros((2, 3), dtype=np.float32),
        "labels": np.asarray([0, 1], dtype=np.int64),
        "example_ids": np.asarray(["e0", "e1"]),
        "question_ids": np.asarray(question_ids),
        "model_name": np.asarray("org/model"),
        "model_revision": np.asarray("a" * 40),
        "feature_schema_version": np.asarray(schema_version),
        "dataset_sha256": np.asarray("b" * 64),
        "code_revision": np.asarray("c" * 40),
        "code_dirty": np.asarray(False),
        "chat_template_used": np.asarray(True),
        "chat_template_sha256": np.asarray("d" * 64),
        "extraction_config_sha256": np.asarray("e" * 64),
    }
    if schema_version == "3":
        arrays.update(
            {
                "original_token_counts": original_counts,
                "token_counts": retained_counts,
                "truncated": np.asarray([truncated, False]),
                "token_spans_json": np.asarray(
                    ['{"answer": [1, 4]}', '{"answer": [1, 4]}']
                ),
                "pooling_mode": np.asarray("mean"),
                "requested_views_json": np.asarray('["answer"]'),
                "max_length": np.asarray(8),
                "allow_truncation": np.asarray(allow_truncation),
                "missing_view_policy": np.asarray("error"),
                "require_model_generated": np.asarray(True),
                "dropped_example_ids": np.asarray(""),
                "resolved_device": np.asarray("cpu"),
                "model_parameter_device": np.asarray("cpu"),
                "model_parameter_dtype": np.asarray("torch.float32"),
                "feature_dtype": np.asarray("float32"),
                "layer_index_semantics": np.asarray(
                    "zero_based_transformer_block_output"
                ),
            }
        )
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / filename, **arrays)


def test_final_feature_validation_requires_auditable_nontruncated_schema(
    tmp_path: Path,
) -> None:
    model = {
        "name": "model",
        "model_id": "org/model",
        "model_revision": "a" * 40,
    }
    valid = tmp_path / "valid"
    _write_feature_bundle(valid, schema_version="3")
    _validate_feature_directory(
        valid, model, "task", 1, ("train",), final_protocol=True
    )

    legacy = tmp_path / "legacy"
    _write_feature_bundle(legacy, schema_version="2")
    with pytest.raises(ValueError, match="schema version 3"):
        _validate_feature_directory(
            legacy, model, "task", 1, ("train",), final_protocol=True
        )

    truncated = tmp_path / "truncated"
    _write_feature_bundle(
        truncated,
        schema_version="3",
        allow_truncation=True,
        truncated=True,
    )
    with pytest.raises(ValueError, match="permits or contains truncation"):
        _validate_feature_directory(
            truncated, model, "task", 1, ("train",), final_protocol=True
        )


def test_feature_validation_requires_identical_examples_across_layers(
    tmp_path: Path,
) -> None:
    model = {
        "name": "model",
        "model_id": "org/model",
        "model_revision": "a" * 40,
    }
    directory = tmp_path / "mismatched_layers"
    _write_feature_bundle(directory, schema_version="3")
    _write_feature_bundle(
        directory,
        schema_version="3",
        filename="train_layer1.npz",
        question_ids=("g0", "different-group"),
    )

    with pytest.raises(ValueError, match="disagrees with other train layer bundles"):
        _validate_feature_directory(
            directory, model, "task", 1, ("train",), final_protocol=True
        )
