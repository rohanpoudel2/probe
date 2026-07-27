from __future__ import annotations

import json

import pytest

from cli.build_reference_traffic_scenarios import build_reference_scenarios
from data.reference_traffic import (
    REFERENCE_LABEL_SOURCE,
    REFERENCE_PROTOCOL,
    build_reference_annotations,
)
from tasks.reference_traffic import ReferenceTrafficTask


def _rollout(
    rollout_id: str,
    group_id: str,
    split: str,
    *,
    capped: bool = False,
) -> dict:
    prompt_messages = [{"role": "user", "content": f"Question {group_id}"}]
    response = f"Answer {group_id}"
    return {
        "rollout_id": rollout_id,
        "scenario_id": f"scenario-{group_id}",
        "group_id": group_id,
        "question_id": group_id,
        "task_family": "reference_traffic",
        "condition": "reference_prompt",
        "protocol_split": split,
        "prompt": prompt_messages[0]["content"],
        "prompt_messages": prompt_messages,
        "messages": [
            *prompt_messages,
            {"role": "assistant", "content": response},
        ],
        "response_text": response,
        "assistant_response": response,
        "model_id": "org/model",
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "data_origin": "on_policy_generation",
        "generated_by_model": True,
        "generation": {
            "response_token_count": 32 if capped else 5,
            "max_new_tokens": 32,
            "stop_reason": "max_new_tokens" if capped else "eos_token",
        },
    }


def test_reference_annotations_encode_membership_without_semantic_labels() -> None:
    annotations, report = build_reference_annotations(
        [
            _rollout("cal-0", "cal-group", "calibration"),
            _rollout("test-0", "test-group", "test"),
        ]
    )
    assert report["semantic_labels_assigned"] is False
    assert report["split_counts"] == {"calibration": 1, "test": 1}
    assert {row["label_source"] for row in annotations} == {
        REFERENCE_LABEL_SOURCE
    }
    assert {row["annotation_protocol"] for row in annotations} == {
        REFERENCE_PROTOCOL
    }
    assert all(
        row["metadata"]["semantic_negative_label"] is False for row in annotations
    )


def test_reference_scenario_builder_freezes_disjoint_partitions() -> None:
    rows = [
        (
            index + 1,
            {
                "conversation": [
                    {"role": "user", "content": f"Natural prompt number {index}"}
                ],
                "conversation_hash": f"conversation-{index}",
                "language": "English",
                "toxic": False,
                "redacted": False,
                "openai_moderation": {"flagged": False},
            },
        )
        for index in range(6)
    ]
    scenarios, counts = build_reference_scenarios(
        rows,
        source_repo="org/reference",
        source_revision="c" * 40,
        source_split="train",
        raw_file_sha256="d" * 64,
        calibration_scenarios=3,
        holdout_scenarios=2,
        min_chars=10,
        max_chars=100,
        selection_seed=7,
    )
    assert counts["selected_calibration"] == 3
    assert counts["selected_holdout"] == 2
    assert [row["protocol_split"] for row in scenarios] == [
        "calibration",
        "calibration",
        "calibration",
        "test",
        "test",
    ]
    assert len({row["group_id"] for row in scenarios}) == 5


def test_reference_annotations_reject_capped_or_repeated_groups() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        build_reference_annotations(
            [_rollout("cal-0", "cal-group", "calibration", capped=True)]
        )
    with pytest.raises(ValueError, match="exactly one rollout"):
        build_reference_annotations(
            [
                _rollout("cal-0", "shared", "calibration"),
                _rollout("test-0", "shared", "test"),
            ]
        )
    with pytest.raises(ValueError, match="both calibration and test"):
        build_reference_annotations(
            [_rollout("cal-0", "cal-group", "calibration")]
        )


def test_reference_task_loads_frozen_merged_records(tmp_path) -> None:
    rollouts = [
        _rollout("cal-0", "cal-group", "calibration"),
        _rollout("test-0", "test-group", "test"),
    ]
    annotations, _ = build_reference_annotations(rollouts)
    merged = []
    for rollout, annotation in zip(rollouts, annotations):
        merged.append(
            {
                **rollout,
                "example_id": rollout["rollout_id"],
                "label": annotation["label"],
                "label_source": annotation["label_source"],
                "annotation_protocol": annotation["annotation_protocol"],
                "annotation_metadata": annotation["metadata"],
            }
        )
    path = tmp_path / "reference.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged),
        encoding="utf-8",
    )
    examples = ReferenceTrafficTask().load(str(path))
    assert [example.label for example in examples] == [0, 0]
    assert [example.metadata["protocol_split"] for example in examples] == [
        "calibration",
        "test",
    ]
