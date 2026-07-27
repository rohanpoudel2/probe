from __future__ import annotations

from typing import List, Optional

from data.schema import TaskExample
from tasks.base import BehaviorTask, TaskSpec
from tasks.jsonl_utils import read_jsonl, require_fields, rollout_metadata


class SycophancyTask(BehaviorTask):
    spec = TaskSpec(
        name="sycophancy",
        label_semantics={0: "non_sycophantic", 1: "sycophantic"},
        grouped_split_key="question_id",
    )

    def load(self, path: Optional[str] = None) -> List[TaskExample]:
        if path is None:
            raise ValueError("SycophancyTask.load requires a JSONL path.")

        rows = read_jsonl(path)
        examples: List[TaskExample] = []
        for idx, row in enumerate(rows):
            require_fields(row, ["prompt", "label"], path)
            context = row.get("pressure_context") or row.get("context") or row.get("user_belief")
            answer = row.get("final_answer") or row.get("assistant_response")
            reasoning = row.get("chain_of_thought") or row.get("reasoning")
            segments = {}
            if context:
                segments["pressure_context"] = context
            segments["prompt"] = row["prompt"]
            if reasoning:
                segments["reasoning"] = reasoning
            if answer:
                segments["answer"] = answer

            examples.append(
                TaskExample(
                    example_id=str(row.get("example_id", idx)),
                    task_family="sycophancy",
                    prompt=row["prompt"],
                    label=int(row["label"]),
                    question_id=row.get("question_id") or row.get("group_id") or str(row.get("example_id", idx)),
                    condition=row.get("condition", "agreement"),
                    context=context,
                    assistant_response=row.get("assistant_response"),
                    final_answer=row.get("final_answer"),
                    chain_of_thought=reasoning,
                    metadata=rollout_metadata(
                        row,
                        topic=row.get("topic"),
                        agreement_target=row.get("agreement_target"),
                    ),
                    messages=row.get("messages", []),
                    segments=segments,
                )
            )
        return examples
