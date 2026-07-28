import pytest
import numpy as np
import yaml
from pathlib import Path

from cli.validate_multimodel_config import (
    _validate_frozen_artifact_provenance,
    _validate_feature_directory,
    validate_config,
)


def _config():
    return {
        "protocol_version": "frontier-v1",
        "protocol_stage": "selection",
        "execution_mode": "selection",
        "results_dir": "results",
        "views": "answer",
        "layers": "all",
        "probes": "P1_logistic",
        "max_reference_alert_rate": 0.01,
        "min_reference_groups": 1000,
        "k_values": "1,4,8",
        "selection_k": 8,
        "seeds": 10,
        "balance_modes": "balanced",
        "bootstrap_samples": 1000,
        "run_black_box_baselines": False,
        "run_falsification_suite": False,
        "falsification_registry": "experiments/protocol/falsification_registry.yaml",
        "claim_gates": {
            "primary_effect": "mean_unseen_behavior_auew_uplift",
            "primary_probe": "P8_citm",
            "primary_k": 8,
            "early_warning_basis": "assistant_response_tokens_v1",
            "matched_visible_prefix_basis": "assistant_response_characters_v1",
            "primary_inference": "paired_hierarchical_auew_bootstrap_v1",
            "auew_weighting": "early_linear_v1",
            "reference_holdout_check": "selected_system_interval_not_above_budget_v1",
            "hard_negative_check": "fpr_noninferiority_and_pairwise_ordering_v1",
            "hard_negative_fpr_noninferiority_margin": 0.01,
            "pairwise_order_accuracy_floor": 0.5,
            "multiplicity_control": "holm_global",
        },
        "task_pairs": [{"source_task": "a", "target_task": "b"}],
        "models": [
            {
                "name": "model-a",
                "model_id": "org/model-a",
                "model_revision": "abcdef1234567",
                "family": "family-a",
                "feature_dirs": {"a": "features/a", "b": "features/b"},
                "reference_feature_dir": "features/reference",
            }
        ],
    }


def test_selection_config_accepts_pinned_model() -> None:
    validate_config(_config(), check_paths=False, final_protocol=False)


def test_final_config_requires_three_families_and_preregistration() -> None:
    config = _config()
    config["protocol_stage"] = "frozen"
    config["execution_mode"] = "confirmatory"
    config["min_reference_groups"] = 10_000
    config["k_values"] = "1,2,4,8,16,32"
    config["reference_traffic_protocol"] = {
        "protocol": "reference-traffic-v1",
        "scope": "per_monitored_model_revision",
        "calibration_method": "split_conformal_upper_tail_v1",
        "max_reference_alert_rate": 0.01,
        "selection_seed": 9173,
        "calibration_groups": 10_000,
        "holdout_groups": 10_000,
        "semantic_labels": False,
    }
    with pytest.raises(ValueError, match="three genuinely different"):
        validate_config(config, check_paths=False, final_protocol=True)


def test_moving_model_revision_is_rejected() -> None:
    config = _config()
    config["models"][0]["model_revision"] = "main"
    with pytest.raises(ValueError, match="pinned"):
        validate_config(config, check_paths=False, final_protocol=False)


def test_selection_validates_the_registered_embedding_lock() -> None:
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


def test_selection_validates_the_registered_reference_protocol() -> None:
    config = _config()
    config["reference_traffic_protocol"] = {
        "protocol": "reference-traffic-v1",
        "scope": "per_monitored_model_revision",
        "calibration_method": "split_conformal_upper_tail_v1",
        "max_reference_alert_rate": 0.01,
        "selection_seed": 9173,
        "calibration_groups": 1000,
        "holdout_groups": 1000,
        "semantic_labels": False,
    }
    validate_config(config, check_paths=False, final_protocol=False)

    config["reference_traffic_protocol"]["holdout_groups"] = 0
    with pytest.raises(ValueError, match="untouched holdout"):
        validate_config(config, check_paths=False, final_protocol=False)


def test_protocol_stage_and_execution_mode_cannot_diverge() -> None:
    config = _config()
    config["execution_mode"] = "confirmatory"
    with pytest.raises(ValueError, match="requires execution_mode=selection"):
        validate_config(config, check_paths=False, final_protocol=False)


def test_frozen_artifact_provenance_rejects_comparison_tampering(
    tmp_path: Path,
) -> None:
    registry_sha256 = "a" * 64
    provenance = {
        "selection_file": "results/selection/task_primary_source_systems.csv",
        "selection_file_sha256": "b" * 64,
        "early_warning_selection_file": (
            "results/selection/early_warning_source_selection.csv"
        ),
        "early_warning_selection_file_sha256": "d" * 64,
        "base_config_sha256": "c" * 64,
        "selection_k": 8,
        "registry_sha256": registry_sha256,
    }
    primary = {
        "schema_version": "primary-v1",
        "selection_provenance": provenance,
        "comparisons": [],
    }
    falsification = {
        "schema_version": "falsification-v1",
        "selection_provenance": provenance,
        "comparisons": [],
    }
    primary_path = tmp_path / "primary.yaml"
    falsification_path = tmp_path / "falsification.yaml"
    primary_path.write_text(
        yaml.safe_dump(primary, sort_keys=False),
        encoding="utf-8",
    )
    falsification_path.write_text(
        yaml.safe_dump(falsification, sort_keys=False),
        encoding="utf-8",
    )
    import hashlib

    cfg = {
        "selection_k": 8,
        "selection_provenance": provenance,
        "registered_artifact_sha256": {
            "comparisons_file": hashlib.sha256(
                primary_path.read_bytes()
            ).hexdigest(),
            "falsification_comparisons_file": hashlib.sha256(
                falsification_path.read_bytes()
            ).hexdigest(),
        },
    }
    _validate_frozen_artifact_provenance(
        cfg,
        registry_sha256=registry_sha256,
        comparisons_path=primary_path,
        comparisons_config=primary,
        falsification_comparisons_path=falsification_path,
        falsification_comparisons_config=falsification,
    )

    primary_path.write_text(
        primary_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest hashes"):
        _validate_frozen_artifact_provenance(
            cfg,
            registry_sha256=registry_sha256,
            comparisons_path=primary_path,
            comparisons_config=primary,
            falsification_comparisons_path=falsification_path,
            falsification_comparisons_config=falsification,
        )


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
        "annotation_outcome_class": np.asarray(
            ["correct_non_target", "target_aligned"]
        ),
        "model_name": np.asarray("org/model"),
        "model_revision": np.asarray("a" * 40),
        "tokenizer_revision": np.asarray("a" * 40),
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

    unsupported = tmp_path / "unsupported"
    _write_feature_bundle(unsupported, schema_version="2")
    with pytest.raises(ValueError, match="schema version 3"):
        _validate_feature_directory(
            unsupported, model, "task", 1, ("train",), final_protocol=True
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


def test_feature_validation_requires_manifest_activation_views(
    tmp_path: Path,
) -> None:
    model = {
        "name": "model",
        "model_id": "org/model",
        "model_revision": "a" * 40,
    }
    directory = tmp_path / "features"
    _write_feature_bundle(directory, schema_version="3")
    with pytest.raises(ValueError, match="lacks requested activation views"):
        _validate_feature_directory(
            directory,
            model,
            "task",
            1,
            ("train",),
            final_protocol=True,
            required_views=("answer", "full_text"),
        )


def test_feature_validation_rejects_stale_dataset_hash(
    tmp_path: Path,
) -> None:
    model = {
        "name": "model",
        "model_id": "org/model",
        "model_revision": "a" * 40,
    }
    directory = tmp_path / "features"
    _write_feature_bundle(directory, schema_version="3")
    with pytest.raises(ValueError, match="different labeled dataset"):
        _validate_feature_directory(
            directory,
            model,
            "task",
            1,
            ("train",),
            final_protocol=True,
            required_views=("answer",),
            expected_dataset_sha256="f" * 64,
        )
