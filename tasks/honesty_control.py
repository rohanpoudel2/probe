from __future__ import annotations

from typing import List, Optional

from data.schema import TaskExample
from tasks.base import BehaviorTask, TaskSpec
from tasks.jsonl_utils import read_jsonl, require_fields, rollout_metadata


class HonestyControlTask(BehaviorTask):
    spec = TaskSpec(
        name="honesty_control",
        label_semantics={0: "honest_or_control", 1: "dishonest_reporting"},
        grouped_split_key="question_id",
    )

    def load(self, path: Optional[str] = None) -> List[TaskExample]:
        if path is None:
            raise ValueError("HonestyControlTask.load requires a JSONL path.")

        rows = read_jsonl(path)
        examples: List[TaskExample] = []
        for idx, row in enumerate(rows):
            require_fields(row, ["prompt", "label"], path)
            scenario_metadata = {}
            if isinstance(row.get("metadata"), dict):
                candidate_metadata = row["metadata"].get("scenario_metadata")
                if isinstance(candidate_metadata, dict):
                    scenario_metadata = candidate_metadata
            context = (
                row.get("honesty_context")
                or row.get("context")
                or row.get("instruction")
                or scenario_metadata.get("honesty_context")
            )
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
                    condition=row.get("condition", "neutral_control"),
                    context=context,
                    assistant_response=row.get("assistant_response"),
                    final_answer=row.get("final_answer") or row.get("reported_answer"),
                    chain_of_thought=reasoning,
                    metadata=rollout_metadata(
                        row,
                        ground_truth_answer=(
                            row.get("ground_truth_answer")
                            if "ground_truth_answer" in row
                            else scenario_metadata.get("ground_truth_answer")
                            or scenario_metadata.get("gold_answer")
                        ),
                        known_answer=(
                            row.get("known_answer")
                            if "known_answer" in row
                            else scenario_metadata.get("known_answer")
                        ),
                    ),
                    messages=row.get("messages", []),
                    segments=segments,
                )
            )
        return examples
