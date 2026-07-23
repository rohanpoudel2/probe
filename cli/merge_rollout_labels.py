from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from data.benign_audit import (
    AUDIT_LABEL_SOURCE,
    AUDIT_PROTOCOL,
    validate_accepted_audit_metadata,
)
from data.benign_screening import (
    ANNOTATION_PROTOCOL as BENIGN_ANNOTATION_PROTOCOL,
    LABEL_SOURCE as BENIGN_LABEL_SOURCE,
    screened_text_sha256,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from err
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _annotation_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    for row in rows:
        rollout_id = str(row.get("rollout_id", "")).strip()
        if not rollout_id:
            raise ValueError("Every annotation requires rollout_id")
        if rollout_id in annotations:
            raise ValueError(f"Duplicate annotation for {rollout_id}")
        label = row.get("label")
        excluded = row.get("excluded") is True
        if excluded:
            if label is not None or not str(row.get("exclude_reason", "")).strip():
                raise ValueError(
                    f"Excluded annotation {rollout_id} requires label=null and exclude_reason"
                )
        elif label not in {0, 1}:
            raise ValueError(f"Annotation {rollout_id} requires binary label 0 or 1")
        for field in ("label_source", "annotation_protocol"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Annotation {rollout_id} requires non-empty {field}")
        annotations[rollout_id] = row
    return annotations


def _last_user_content(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError("Rollout has no user message")


def _validate_benign_annotation(
    rollout: dict[str, Any], annotation: dict[str, Any]
) -> None:
    rollout_id = rollout.get("rollout_id")
    if annotation.get("label") != 0:
        raise ValueError(
            f"Benign candidate {rollout_id} must be excluded unless its accepted label is 0"
        )
    if (
        annotation.get("label_source") == AUDIT_LABEL_SOURCE
        and annotation.get("annotation_protocol") == AUDIT_PROTOCOL
    ):
        validate_accepted_audit_metadata(rollout, annotation.get("metadata"))
        return
    if (
        annotation.get("label_source") != BENIGN_LABEL_SOURCE
        or annotation.get("annotation_protocol") != BENIGN_ANNOTATION_PROTOCOL
    ):
        raise ValueError(
            f"Benign candidate {rollout_id} lacks the frozen screening protocol"
        )
    metadata = annotation.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Benign candidate {rollout_id} lacks screening metadata")
    n_raters = metadata.get("n_independent_raters")
    if (
        not isinstance(n_raters, int)
        or isinstance(n_raters, bool)
        or n_raters < 2
        or metadata.get("unanimous_eligible") is not True
    ):
        raise ValueError(
            f"Benign candidate {rollout_id} lacks unanimous two-rater screening"
        )
    if metadata.get("screened_text_sha256") != screened_text_sha256(rollout):
        raise ValueError(
            f"Benign candidate {rollout_id} has a stale screening text hash"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge independently adjudicated labels into rollouts"
    )
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rollouts = _read_jsonl(Path(args.rollouts))
    annotations = _annotation_map(_read_jsonl(Path(args.annotations)))
    rollout_ids = [str(row.get("rollout_id", "")) for row in rollouts]
    if len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("Rollout file contains duplicate rollout_id values")
    missing = sorted(set(rollout_ids).difference(annotations))
    extra = sorted(set(annotations).difference(rollout_ids))
    if missing or extra:
        raise ValueError(
            f"Annotations must match rollouts exactly; missing={missing[:5]}, extra={extra[:5]}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        included = 0
        excluded = 0
        for rollout in rollouts:
            if rollout.get("data_origin") != "on_policy_generation" or not rollout.get(
                "generated_by_model"
            ):
                raise ValueError(
                    f"Rollout {rollout.get('rollout_id')} is not on-policy model output"
                )
            annotation = annotations[str(rollout["rollout_id"])]
            if annotation.get("excluded") is True:
                excluded += 1
                continue
            if rollout.get("task_family") == "benign_calibration":
                _validate_benign_annotation(rollout, annotation)
            final_answer = (
                annotation.get("final_answer")
                or rollout.get("final_answer")
                or rollout["response_text"]
            )
            reasoning = annotation.get("reasoning") or rollout.get("reasoning")
            row = {
                **rollout,
                "example_id": rollout["rollout_id"],
                "question_id": rollout["group_id"],
                "protocol_split": rollout["protocol_split"],
                "prompt": _last_user_content(rollout["prompt_messages"]),
                "assistant_response": rollout["response_text"],
                "final_answer": final_answer,
                "reasoning": reasoning,
                "label": int(annotation["label"]),
                "label_source": annotation["label_source"],
                "annotation_protocol": annotation["annotation_protocol"],
                "annotation_metadata": annotation.get("metadata", {}),
                "eligible_for_main_study": True,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            included += 1
        handle.flush()
        os.fsync(handle.fileno())
    if included == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            "All rollouts were excluded; refusing to create an empty labeled dataset"
        )
    temporary.replace(output)
    print(
        f"saved {included} labeled on-policy rollouts to {output}; excluded {excluded}"
    )


if __name__ == "__main__":
    main()
