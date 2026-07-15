import pytest

from data.rollout_schema import RolloutRecord, ScenarioRecord


def _scenario() -> ScenarioRecord:
    return ScenarioRecord.from_dict(
        {
            "scenario_id": "s1",
            "group_id": "q1",
            "task_family": "sycophancy",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "condition": "user_pressure",
            "protocol_split": "train",
            "source": "fixture",
            "metadata": {"gold_answer": "4"},
        }
    )


def test_scenario_rejects_authored_outcome() -> None:
    with pytest.raises(ValueError, match="authored outcome"):
        ScenarioRecord.from_dict(
            {
                "scenario_id": "s1",
                "group_id": "q1",
                "task_family": "sycophancy",
                "messages": [{"role": "user", "content": "Question"}],
                "condition": "pressure",
                "protocol_split": "train",
                "source": "fixture",
                "label": 1,
            }
        )


def test_rollout_requires_pinned_revision() -> None:
    scenario = _scenario()
    with pytest.raises(ValueError, match="immutable"):
        RolloutRecord(
            rollout_id="r1",
            scenario=scenario,
            response_text="4",
            messages=[*scenario.messages, {"role": "assistant", "content": "4"}],
            model_id="model",
            model_revision="main",
            tokenizer_revision="main",
            seed=0,
            generation={},
            provenance={},
        )


def test_valid_rollout_is_marked_on_policy() -> None:
    scenario = _scenario()
    record = RolloutRecord(
        rollout_id="r1",
        scenario=scenario,
        response_text="4",
        messages=[*scenario.messages, {"role": "assistant", "content": "4"}],
        model_id="model",
        model_revision="abc123",
        tokenizer_revision="abc123",
        seed=0,
        generation={"max_new_tokens": 8},
        provenance={"code_commit": "def456"},
        final_answer="4",
    ).to_dict()
    assert record["data_origin"] == "on_policy_generation"
    assert record["generated_by_model"] is True
