from __future__ import annotations

import pytest

from data.schema import TaskExample
from data.trajectory_schema import build_trajectory_prefix_views
from extraction.task_extractor import TaskActivationExtractor, TaskExtractionConfig


def _stub_extractor(*, allow_truncation: bool) -> TaskActivationExtractor:
    extractor = object.__new__(TaskActivationExtractor)
    extractor.cfg = TaskExtractionConfig(
        model_name="stub",
        layers=[0],
        max_length=4,
        allow_truncation=allow_truncation,
        views=["answer"],
        use_chat_template=False,
        require_model_generated=False,
    )
    extractor._tokenize_raw = lambda segments: {
        "input_ids": [0, 1, 2, 3, 4, 5],
        "token_spans": {
            "prompt": (0, 3),
            "answer": (3, 6),
            "assistant_response": (3, 6),
        },
    }
    return extractor


def _example() -> TaskExample:
    return TaskExample(
        example_id="example-1",
        task_family="stub",
        prompt="prompt",
        assistant_response="answer",
        label=1,
    )


def test_activation_extraction_fails_on_implicit_truncation() -> None:
    extractor = _stub_extractor(allow_truncation=False)
    with pytest.raises(ValueError, match="truncation is prohibited"):
        extractor._tokenize_segments(_example())


def test_extraction_config_rejects_nonpositive_max_length() -> None:
    with pytest.raises(ValueError, match="max_length must be a positive integer"):
        TaskExtractionConfig(model_name="stub", layers=[0], max_length=0)


def test_explicit_truncation_records_counts_and_shifted_span() -> None:
    extractor = _stub_extractor(allow_truncation=True)
    encoded = extractor._tokenize_segments(_example())
    assert encoded["original_token_count"] == 6
    assert encoded["token_count"] == 4
    assert encoded["truncated"] is True
    assert encoded["token_spans"]["answer"] == (1, 4)


def test_trajectory_views_are_built_by_percentile() -> None:
    assert build_trajectory_prefix_views([10, 33, 100], token_count=17) == [
        ("trajectory_prefix_p10", 2),
        ("trajectory_prefix_p33", 6),
        ("trajectory_prefix_p100", 17),
    ]


def test_requested_output_views_include_prefix_views() -> None:
    extractor = _stub_extractor(allow_truncation=False)
    extractor.cfg.trajectory_prefix_percentiles = [25, 50]
    extractor.cfg.views = ["answer"]
    extractor.cfg.max_length = 10
    encoded = extractor._tokenize_segments(_example())
    assert "trajectory_prefix_p25" in encoded["token_spans"]
    assert "trajectory_prefix_p50" in encoded["token_spans"]
    assert encoded["token_spans"]["trajectory_prompt_end"] == (2, 3)
    assert encoded["token_spans"]["trajectory_prefix_p25"] == (3, 4)
