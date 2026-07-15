import pytest
from pathlib import Path

from cli.common import load_yaml
from cli.validate_multimodel_config import validate_config


def _config():
    return {
        "protocol_version": "frontier-v1",
        "protocol_stage": "pilot",
        "results_dir": "results",
        "max_fpr": 0.01,
        "min_calibration_negatives": 1000,
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
