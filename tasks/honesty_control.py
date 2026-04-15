from __future__ import annotations

from typing import List, Optional

from data.schema import TaskExample
from tasks.base import BehaviorTask, TaskSpec
from tasks.jsonl_utils import read_jsonl, require_fields


class HonestyControlTask(BehaviorTask):
    spec = TaskSpec(
        name="honesty_control",
        primary_metric="auroc",
        label_semantics={0: "honest_or_control", 1: "dishonest_reporting"},
        grouped_split_key="question_id",
        default_spans=["full_text", "reasoning", "answer"],
        notes="Auxiliary honesty-vs-accuracy control family, intended for MASK-style lying benchmarks.",
    )

    def load(self, path: Optional[str] = None) -> List[TaskExample]:
        if path is None:
            raise ValueError("HonestyControlTask.load requires a JSONL path in Phase 1.")

        rows = read_jsonl(path)
        examples: List[TaskExample] = []
        for idx, row in enumerate(rows):
            require_fields(row, ["prompt", "label"], path)
            context = row.get("honesty_context") or row.get("context") or row.get("instruction")
            reasoning = row.get("chain_of_thought") or row.get("reasoning") or row.get("explanation")
            answer = row.get("final_answer") or row.get("assistant_response") or row.get("reported_answer")
            segments = {}
            if context:
                segments["context"] = context
            segments["prompt"] = row["prompt"]
            if reasoning:
                segments["reasoning"] = reasoning
            if answer:
                segments["answer"] = answer

            examples.append(
                TaskExample(
                    example_id=str(row.get("example_id", idx)),
                    task_family="honesty_control",
                    prompt=row["prompt"],
                    label=int(row["label"]),
                    question_id=row.get("question_id") or str(row.get("example_id", idx)),
                    condition=row.get("condition", "honesty"),
                    context=context,
                    assistant_response=row.get("assistant_response"),
                    final_answer=row.get("final_answer") or row.get("reported_answer"),
                    chain_of_thought=reasoning,
                    metadata={
                        "source": row.get("source", "jsonl"),
                        "ground_truth_answer": row.get("ground_truth_answer"),
                        "known_answer": row.get("known_answer"),
                    },
                    segments=segments,
                )
            )
        return examples
