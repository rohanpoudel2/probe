from __future__ import annotations

from typing import List, Optional

from data.schema import TaskExample
from tasks.base import BehaviorTask, TaskSpec
from tasks.jsonl_utils import read_jsonl, require_fields


class MotivatedReasoningTask(BehaviorTask):
    spec = TaskSpec(
        name="motivated_reasoning",
        primary_metric="auroc",
        label_semantics={0: "unbiased_or_control", 1: "motivated_reasoning"},
        grouped_split_key="question_id",
        default_spans=["full_text", "hint_context", "reasoning", "reasoning_early", "reasoning_mid", "reasoning_late", "answer"],
        notes="Transfer family with explicit support for before-CoT and after-CoT analysis.",
    )

    def load(self, path: Optional[str] = None) -> List[TaskExample]:
        if path is None:
            raise ValueError("MotivatedReasoningTask.load requires a JSONL path in Phase 1.")

        rows = read_jsonl(path)
        examples: List[TaskExample] = []
        for idx, row in enumerate(rows):
            require_fields(row, ["prompt", "label"], path)
            hint = row.get("hint_context") or row.get("context") or row.get("biasing_hint")
            reasoning = row.get("chain_of_thought") or row.get("reasoning")
            answer = row.get("final_answer") or row.get("assistant_response")
            segments = {}
            if hint:
                segments["hint_context"] = hint
            segments["prompt"] = row["prompt"]
            if reasoning:
                segments["reasoning"] = reasoning
            if answer:
                segments["answer"] = answer

            examples.append(
                TaskExample(
                    example_id=str(row.get("example_id", idx)),
                    task_family="motivated_reasoning",
                    prompt=row["prompt"],
                    label=int(row["label"]),
                    question_id=row.get("question_id") or str(row.get("example_id", idx)),
                    condition=row.get("condition", "hinted"),
                    context=hint,
                    assistant_response=row.get("assistant_response"),
                    final_answer=row.get("final_answer"),
                    chain_of_thought=reasoning,
                    metadata={
                        "source": row.get("source", "jsonl"),
                        "target_conclusion": row.get("target_conclusion"),
                    },
                    segments=segments,
                )
            )
        return examples
