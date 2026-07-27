import torch

from cli.audit_rollout_dataset import audit_rows
from data.reference_traffic import (
    REFERENCE_LABEL_SOURCE,
    REFERENCE_PROTOCOL,
    reference_text_sha256,
)
from data.generation_confidence import build_generation_confidence_trace


def _row(rollout_id: str, scenario_id: str, group_id: str, condition: str, label: int):
    prompt = [{"role": "user", "content": f"Prompt for {group_id} {condition}"}]
    response = f"response {label}"
    return {
        "rollout_id": rollout_id,
        "scenario_id": scenario_id,
        "group_id": group_id,
        "question_id": group_id,
        "condition": condition,
        "protocol_split": "train" if group_id == "q0" else "test",
        "prompt_messages": prompt,
        "messages": [*prompt, {"role": "assistant", "content": response}],
        "response_text": response,
        "label": label,
        "label_source": "executable_task_rule",
        "annotation_protocol": "fixture-v1",
        "model_id": "model",
        "model_revision": "abcdef1234567",
        "tokenizer_revision": "abcdef1234567",
        "provenance": {
            "code_commit": "1234567abcdef",
            "code_dirty": False,
            "chat_template_sha256": "a" * 64,
            "scenario_file_sha256": "b" * 64,
        },
        "data_origin": "on_policy_generation",
        "generated_by_model": True,
    }


def test_rollout_audit_accepts_matched_on_policy_groups() -> None:
    rows = [
        _row("r0", "s0", "q0", "pressure", 1),
        _row("r1", "s1", "q0", "neutral", 0),
        _row("r2", "s2", "q1", "pressure", 1),
        _row("r3", "s3", "q1", "neutral", 0),
    ]
    report = audit_rows(rows)
    assert report["status"] == "pass"
    assert report["n_matched_groups"] == 2


def test_rollout_audit_rejects_authored_or_unmatched_data() -> None:
    row = _row("r0", "s0", "q0", "pressure", 1)
    row["data_origin"] = "authored_synthetic_debug"
    report = audit_rows([row])
    assert report["status"] == "fail"
    assert any("on-policy" in error for error in report["errors"])


def test_rollout_audit_can_require_aligned_generation_confidence() -> None:
    rows = [
        _row("r0", "s0", "q0", "pressure", 1),
        _row("r1", "s1", "q0", "neutral", 0),
    ]
    missing = audit_rows(rows, require_confidence_trace=True)
    assert missing["status"] == "fail"
    for row in rows:
        token_ids = [1, 2]
        row["generation"] = {
            "response_token_count": len(token_ids),
            "response_token_ids": token_ids,
            "confidence_trace": build_generation_confidence_trace(
                [torch.tensor([0.0, 2.0, 1.0]), torch.tensor([0.0, 1.0, 2.0])],
                token_ids,
            ),
        }
    report = audit_rows(rows, require_confidence_trace=True)
    assert report["status"] == "pass"
    assert report["n_valid_generation_confidence_traces"] == 2


def _as_reference(row: dict, split: str) -> dict:
    row["task_family"] = "reference_traffic"
    row["condition"] = "reference_prompt"
    row["protocol_split"] = split
    row["label_source"] = REFERENCE_LABEL_SOURCE
    row["annotation_protocol"] = REFERENCE_PROTOCOL
    row["generation"] = {
        "response_token_count": 2,
        "max_new_tokens": 32,
        "stop_reason": "eos_token",
    }
    row["annotation_metadata"] = {
        "reference_membership_only": True,
        "semantic_negative_label": False,
        "reference_partition": split,
        "reference_text_sha256": reference_text_sha256(row),
    }
    return row


def test_reference_traffic_mode_accepts_disjoint_membership_groups() -> None:
    rows = [
        _as_reference(_row("r0", "s0", "q0", "reference", 0), "calibration"),
        _as_reference(_row("r1", "s1", "q1", "reference", 0), "test"),
    ]
    report = audit_rows(rows, reference_traffic=True)
    assert report["status"] == "pass"


def test_reference_traffic_mode_rejects_repeated_groups() -> None:
    rows = [
        _as_reference(_row("r0", "s0", "q0", "reference", 0), "calibration"),
        _as_reference(_row("r1", "s1", "q0", "reference", 0), "calibration"),
    ]
    report = audit_rows(rows, reference_traffic=True)
    assert report["status"] == "fail"
    assert any("exactly one rollout" in error for error in report["errors"])
