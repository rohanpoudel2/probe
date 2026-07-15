from __future__ import annotations

from typing import List, Optional

from data.monitorbench import (
    DEFAULT_MONITORBENCH_ADAPTER,
    load_monitorbench_adapter,
    validate_normalized_monitorbench_row,
)
from data.schema import TaskExample
from tasks.base import BehaviorTask, TaskSpec
from tasks.jsonl_utils import read_jsonl, require_fields, rollout_metadata


class CoTDistortionTask(BehaviorTask):
    spec = TaskSpec(
        name="cot_distortion",
        primary_metric="recall_at_frozen_fpr",
        label_semantics={
            0: "official_target_outcome_not_verified",
            1: "official_target_outcome_verified",
        },
        grouped_split_key="question_id",
        default_spans=[
            "full_text",
            "reasoning",
            "reasoning_early",
            "reasoning_mid",
            "reasoning_late",
            "pre_answer",
            "answer",
        ],
        evaluation_only=True,
        unavailable_baselines={
            "B4_output_confidence_logistic": (
                "Official MonitorBench .tested.jsonl files do not serialize the "
                "per-token distributions required for generation-confidence features."
            )
        },
        notes=(
            "Test-only official MonitorBench derived outcome family. Labels are official "
            "target-outcome verifier results, not CoT-unfaithfulness labels or the official "
            "MonitorBench monitorability metric."
        ),
    )

    def load(self, path: Optional[str] = None) -> List[TaskExample]:
        if path is None:
            raise ValueError("CoTDistortionTask.load requires a JSONL path in Phase 1.")

        adapter, adapter_sha256 = load_monitorbench_adapter(
            DEFAULT_MONITORBENCH_ADAPTER
        )
        rows = read_jsonl(path)
        examples: List[TaskExample] = []
        for idx, row in enumerate(rows):
            require_fields(
                row,
                [
                    "prompt",
                    "label",
                    "task_family",
                    "protocol_split",
                    "condition",
                    "metadata",
                    "prompt_messages",
                    "messages",
                    "response_text",
                    "chain_of_thought",
                    "final_answer",
                ],
                path,
            )
            validate_normalized_monitorbench_row(
                row, adapter=adapter, adapter_sha256=adapter_sha256
            )
            reasoning = row["chain_of_thought"]
            answer = row["final_answer"]
            pre_answer = row.get("pre_answer")
            segments = {"prompt": row["prompt"]}
            segments["reasoning"] = reasoning
            if pre_answer:
                segments["pre_answer"] = pre_answer
            segments["answer"] = answer

            examples.append(
                TaskExample(
                    example_id=str(row.get("example_id") or row["rollout_id"]),
                    task_family="cot_distortion",
                    prompt=row["prompt"],
                    label=int(row["label"]),
                    question_id=row.get("question_id") or row["group_id"],
                    condition=row["condition"],
                    assistant_response=row["response_text"],
                    final_answer=answer,
                    chain_of_thought=reasoning,
                    metadata=rollout_metadata(
                        row,
                        monitorbench=row["metadata"]["monitorbench"],
                    ),
                    segments=segments,
                    messages=row["messages"],
                )
            )
        return examples
