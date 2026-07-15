import numpy as np
import pytest
import torch

from data.generation_confidence import (
    CONFIDENCE_FEATURE_NAMES,
    build_generation_confidence_trace,
    confidence_feature_vector,
    validate_generation_confidence_trace,
)


def _trace():
    scores = [
        torch.tensor([[2.0, 0.0, -1.0]]),
        torch.tensor([[0.0, 1.0, -2.0]]),
        torch.tensor([[-1.0, 0.0, 2.0]]),
    ]
    return build_generation_confidence_trace(scores, torch.tensor([0, 0, 2]))


def test_generation_trace_aligns_selected_tokens_and_distribution_statistics() -> None:
    trace = _trace()
    assert len(trace["selected_token_logprobs"]) == 3
    assert trace["selected_token_is_top1"] == [True, False, True]
    assert all(0.0 <= value <= 1.0 for value in trace["top1_top2_probability_margins"])
    validate_generation_confidence_trace(trace, expected_token_count=3)


def test_confidence_features_are_fixed_dimensional_and_finite() -> None:
    generation = {
        "response_token_count": 3,
        "response_token_ids": [0, 0, 2],
        "confidence_trace": _trace(),
    }
    features = confidence_feature_vector(generation)
    assert features.shape == (len(CONFIDENCE_FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


def test_confidence_trace_hash_detects_tampering() -> None:
    trace = _trace()
    trace["selected_token_logprobs"][0] -= 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_generation_confidence_trace(trace)


def test_confidence_trace_is_bound_to_response_token_ids() -> None:
    trace = _trace()
    with pytest.raises(ValueError, match="does not match response token IDs"):
        validate_generation_confidence_trace(
            trace,
            expected_token_count=3,
            expected_token_ids=[0, 1, 2],
        )


def test_confidence_features_reject_legacy_rollout_without_trace() -> None:
    with pytest.raises(ValueError, match="confidence_trace"):
        confidence_feature_vector(
            {"response_token_count": 1, "response_token_ids": [1]}
        )
