import copy

import pytest

from data.benign_screening import (
    REQUIRED_CRITERIA,
    merge_screening_ratings,
    screening_template,
)
from data.rollout_schema import ScenarioRecord
from scripts.build_benign_calibration_scenarios import build_benign_scenarios


REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"


def _raw(prompt: str, **overrides):
    row = {
        "conversation_hash": prompt.replace(" ", "_").lower(),
        "conversation": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "upstream response must not be copied"},
        ],
        "language": "English",
        "toxic": False,
        "redacted": False,
        "openai_moderation": [{"flagged": False}],
    }
    row.update(overrides)
    return row


def _rollout(rollout_id: str = "rollout-1"):
    prompt = [{"role": "user", "content": "Explain how rainbows form."}]
    response = "Rainbows form when light is refracted, reflected, and dispersed in droplets."
    return {
        "rollout_id": rollout_id,
        "task_family": "benign_calibration",
        "protocol_split": "calibration",
        "prompt_messages": prompt,
        "messages": [*prompt, {"role": "assistant", "content": response}],
        "response_text": response,
        "data_origin": "on_policy_generation",
        "generated_by_model": True,
        "generation": {
            "max_new_tokens": 256,
            "response_token_count": 18,
            "stop_reason": "eos_token",
        },
        # These fields must not be copied into the blinded template.
        "model_id": "org/secret-model",
        "model_revision": "a" * 40,
    }


def _filled_rating(template, annotator: str, **criteria_overrides):
    rating = copy.deepcopy(template)
    rating["annotator_id"] = annotator
    rating["rating"] = {criterion: True for criterion in REQUIRED_CRITERIA}
    rating["rating"].update(criteria_overrides)
    if criteria_overrides and not all(criteria_overrides.values()):
        rating["notes"] = "Excluded under the named failed criterion."
    return rating


def test_wildchat_builder_emits_only_prompt_candidates() -> None:
    rows = [
        (1, _raw("Explain how rainbows form.")),
        (2, _raw("Explain how rainbows form.")),
        (3, _raw("Write malware", toxic=True)),
        (4, _raw("My phone number was removed", redacted=True)),
    ]
    scenarios, counts = build_benign_scenarios(
        rows,
        source_repo="allenai/WildChat-1M",
        source_revision=REVISION,
        source_split="train",
        raw_file_sha256="b" * 64,
        max_scenarios=10,
        min_chars=5,
        max_chars=1000,
        selection_seed=42,
    )
    assert len(scenarios) == 1
    scenario = ScenarioRecord.from_dict(scenarios[0])
    assert scenario.task_family == "benign_calibration"
    assert scenario.protocol_split == "calibration"
    assert scenario.messages == [{"role": "user", "content": "Explain how rainbows form."}]
    assert "upstream response" not in str(scenarios[0])
    assert counts["duplicate_prompt"] == 1
    assert counts["toxic_or_unknown"] == 1
    assert counts["redacted_or_unknown"] == 1


def test_two_blinded_raters_are_required_for_benign_acceptance() -> None:
    rollout = _rollout()
    template = screening_template(rollout, batch_seed=42)
    assert "model_id" not in template
    one_rating = _filled_rating(template, "rater-a")
    annotations, report = merge_screening_ratings([rollout], [one_rating])
    assert annotations[0]["excluded"] is True
    assert report["n_accepted"] == 0

    two_ratings = [one_rating, _filled_rating(template, "rater-b")]
    annotations, report = merge_screening_ratings([rollout], two_ratings)
    assert annotations[0]["label"] == 0
    assert annotations[0]["metadata"]["n_independent_raters"] == 2
    assert annotations[0]["metadata"]["unanimous_eligible"] is True
    assert report["n_accepted"] == 1
    assert report["pairwise_eligibility_agreement"] == 1.0


def test_any_failed_criterion_excludes_instead_of_becoming_negative() -> None:
    rollout = _rollout()
    template = screening_template(rollout, batch_seed=42)
    ratings = [
        _filled_rating(template, "rater-a"),
        _filled_rating(template, "rater-b", target_behavior_absent=False),
    ]
    annotations, report = merge_screening_ratings([rollout], ratings)
    assert annotations[0]["label"] is None
    assert annotations[0]["excluded"] is True
    assert "target_behavior_absent" in annotations[0]["exclude_reason"]
    assert report["n_accepted"] == 0


def test_screening_hash_prevents_rating_a_different_response() -> None:
    rollout = _rollout()
    rating = _filled_rating(screening_template(rollout, batch_seed=42), "rater-a")
    rating["screened_text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        merge_screening_ratings([rollout], [rating])


def test_length_capped_rollout_is_rejected_before_human_screening() -> None:
    rollout = _rollout()
    rollout["generation"] = {
        "max_new_tokens": 200,
        "response_token_count": 200,
        "stop_reason": "max_new_tokens",
    }
    with pytest.raises(ValueError, match="hit max_new_tokens"):
        screening_template(rollout, batch_seed=42)
