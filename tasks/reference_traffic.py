from __future__ import annotations

from typing import List, Optional

from data.reference_traffic import (
    REFERENCE_LABEL_SOURCE,
    REFERENCE_PROTOCOL,
    validate_reference_annotation_metadata,
)
from data.schema import TaskExample
from tasks.base import BehaviorTask, TaskSpec
from tasks.jsonl_utils import read_jsonl, require_fields, rollout_metadata


class ReferenceTrafficTask(BehaviorTask):
    """Unlabeled on-policy traffic used only to calibrate monitor alert rates."""

    spec = TaskSpec(
        name="reference_traffic",
        label_semantics={0: "reference_distribution_membership_not_semantic_negative"},
        grouped_split_key="question_id",
    )

    def load(self, path: Optional[str] = None) -> List[TaskExample]:
        if not path:
            raise ValueError("ReferenceTrafficTask.load requires a JSONL path")
        examples: List[TaskExample] = []
        for row in read_jsonl(path):
            require_fields(
                row,
                [
                    "example_id",
                    "question_id",
                    "prompt",
                    "assistant_response",
                    "label",
                    "label_source",
                    "annotation_protocol",
                    "annotation_metadata",
                ],
                path,
            )
            if row["label"] != 0:
                raise ValueError(
                    f"Reference row {row['example_id']} must use membership value 0"
                )
            if (
                row["label_source"] != REFERENCE_LABEL_SOURCE
                or row["annotation_protocol"] != REFERENCE_PROTOCOL
            ):
                raise ValueError(
                    f"Reference row {row['example_id']} lacks the frozen contract"
                )
            validate_reference_annotation_metadata(
                row, row.get("annotation_metadata")
            )
            answer = row.get("assistant_response") or row.get("final_answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(
                    f"Reference row {row['example_id']} lacks a response"
                )
            segments = {"prompt": row["prompt"], "answer": answer}
            examples.append(
                TaskExample(
                    example_id=str(row["example_id"]),
                    task_family=self.spec.name,
                    prompt=str(row["prompt"]),
                    label=0,
                    segments=segments,
                    question_id=str(row["question_id"]),
                    condition=str(row.get("condition", "reference_traffic")),
                    messages=row.get("messages") or [],
                    assistant_response=answer,
                    final_answer=row.get("final_answer"),
                    chain_of_thought=row.get("chain_of_thought")
                    or row.get("reasoning"),
                    metadata=rollout_metadata(row),
                )
            )
        if not examples:
            raise ValueError("Reference traffic contains no usable rows")
        groups = [str(example.question_id) for example in examples]
        if len(groups) != len(set(groups)):
            raise ValueError(
                "Reference traffic requires one row per independent group"
            )
        splits = {
            str(example.metadata.get("protocol_split")) for example in examples
        }
        if splits != {"calibration", "test"}:
            raise ValueError(
                "Reference traffic requires calibration and test partitions"
            )
        return examples
