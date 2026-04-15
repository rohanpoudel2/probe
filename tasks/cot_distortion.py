from __future__ import annotations

from typing import List, Optional

from data.schema import TaskExample
from tasks.base import BehaviorTask, TaskSpec
from tasks.jsonl_utils import read_jsonl, require_fields


class CoTDistortionTask(BehaviorTask):
    spec = TaskSpec(
        name="cot_distortion",
        primary_metric="auroc",
        label_semantics={0: "faithful_or_control", 1: "cot_unfaithful"},
        grouped_split_key="question_id",
        default_spans=["full_text", "reasoning", "reasoning_early", "reasoning_mid", "reasoning_late", "pre_answer", "answer"],
        notes="Stress-test family for monitorability and reasoning-report mismatch.",
    )

    def load(self, path: Optional[str] = None) -> List[TaskExample]:
        if path is None:
            raise ValueError("CoTDistortionTask.load requires a JSONL path in Phase 1.")

        rows = read_jsonl(path)
        examples: List[TaskExample] = []
        for idx, row in enumerate(rows):
            require_fields(row, ["prompt", "label"], path)
            reasoning = row.get("chain_of_thought") or row.get("reasoning")
            answer = row.get("final_answer") or row.get("assistant_response")
            pre_answer = row.get("pre_answer")
            segments = {"prompt": row["prompt"]}
            if reasoning:
                segments["reasoning"] = reasoning
            if pre_answer:
                segments["pre_answer"] = pre_answer
            if answer:
                segments["answer"] = answer

            examples.append(
                TaskExample(
                    example_id=str(row.get("example_id", idx)),
                    task_family="cot_distortion",
                    prompt=row["prompt"],
                    label=int(row["label"]),
                    question_id=row.get("question_id") or str(row.get("example_id", idx)),
                    condition=row.get("condition", "faithfulness"),
                    assistant_response=row.get("assistant_response"),
                    final_answer=row.get("final_answer"),
                    chain_of_thought=reasoning,
                    metadata={"source": row.get("source", "jsonl")},
                    segments=segments,
                )
            )
        return examples
