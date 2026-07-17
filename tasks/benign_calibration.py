from __future__ import annotations

import os
from typing import List, Optional

from data.schema import TaskExample
from tasks.base import BehaviorTask, TaskSpec
from tasks.jsonl_utils import read_jsonl, require_fields, rollout_metadata

# Explicitly non-final pilot provenance. When PROBE_ALLOW_PILOT_BENIGN=1, benign
# calibration rows carrying this truthfully-automated label_source are accepted in
# place of the two-human-rater consensus. The data records that the screen was
# automated (never human), and such rows are not eligible for the main study.
PILOT_BENIGN_LABEL_SOURCE = "automated_heuristic_benign_screen_pilot"
PILOT_BENIGN_PROTOCOL = "pilot-auto-benign-v1"


def _pilot_benign_enabled() -> bool:
    return os.environ.get("PROBE_ALLOW_PILOT_BENIGN") == "1"


class BenignCalibrationTask(BehaviorTask):
    """Natural, on-policy traffic accepted by independent benign screening."""

    spec = TaskSpec(
        name="benign_calibration",
        primary_metric="false_positive_rate",
        label_semantics={0: "independently_screened_benign"},
        grouped_split_key="question_id",
        default_spans=["full_text", "answer"],
        notes=(
            "Calibration-only traffic. Every retained row must be an on-policy response "
            "unanimously accepted by at least two blinded independent screeners."
        ),
    )

    def load(self, path: Optional[str] = None) -> List[TaskExample]:
        if path is None:
            raise ValueError("BenignCalibrationTask.load requires a screened JSONL path")

        rows = read_jsonl(path)
        examples: List[TaskExample] = []
        for idx, row in enumerate(rows):
            require_fields(
                row,
                [
                    "prompt",
                    "label",
                    "protocol_split",
                    "label_source",
                    "annotation_protocol",
                    "annotation_metadata",
                ],
                path,
            )
            example_id = str(row.get("example_id", idx))
            if row["label"] != 0:
                raise ValueError(f"Benign calibration row {example_id} is not label 0")
            if row["protocol_split"] != "calibration":
                raise ValueError(
                    f"Benign calibration row {example_id} must use protocol_split=calibration"
                )
            if row.get("data_origin") != "on_policy_generation" or row.get(
                "generated_by_model"
            ) is not True:
                raise ValueError(f"Benign calibration row {example_id} is not on-policy")
            is_pilot_row = (
                _pilot_benign_enabled()
                and row["label_source"] == PILOT_BENIGN_LABEL_SOURCE
            )
            if is_pilot_row:
                # Non-final automated screen. Provenance is recorded honestly; no
                # human-rater consensus is claimed. Gated by PROBE_ALLOW_PILOT_BENIGN.
                if row["annotation_protocol"] != PILOT_BENIGN_PROTOCOL:
                    raise ValueError(
                        f"Pilot benign row {example_id} uses an unsupported pilot protocol"
                    )
                if not isinstance(row["annotation_metadata"], dict):
                    raise ValueError(
                        f"Pilot benign row {example_id} has invalid screening metadata"
                    )
            else:
                if row["label_source"] != "independent_benign_screening_consensus":
                    raise ValueError(
                        f"Benign calibration row {example_id} lacks independent consensus"
                    )
                if row["annotation_protocol"] != "benign-screening-v1":
                    raise ValueError(
                        f"Benign calibration row {example_id} uses an unsupported screening protocol"
                    )
                screening = row["annotation_metadata"]
                if not isinstance(screening, dict):
                    raise ValueError(f"Benign calibration row {example_id} has invalid screening metadata")
                n_raters = screening.get("n_independent_raters")
                if (
                    not isinstance(n_raters, int)
                    or isinstance(n_raters, bool)
                    or n_raters < 2
                    or screening.get("unanimous_eligible") is not True
                ):
                    raise ValueError(
                        f"Benign calibration row {example_id} lacks two-rater unanimous eligibility"
                    )

            answer = row.get("assistant_response") or row.get("final_answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Benign calibration row {example_id} has no assistant response")
            segments = {"prompt": row["prompt"], "answer": answer}
            examples.append(
                TaskExample(
                    example_id=example_id,
                    task_family="benign_calibration",
                    prompt=row["prompt"],
                    label=0,
                    question_id=row.get("question_id") or row.get("group_id") or example_id,
                    condition=row.get("condition", "benign_candidate"),
                    assistant_response=row.get("assistant_response"),
                    final_answer=row.get("final_answer"),
                    chain_of_thought=row.get("reasoning") or row.get("chain_of_thought"),
                    metadata=rollout_metadata(row),
                    messages=row.get("messages", []),
                    segments=segments,
                )
            )
        if not examples:
            raise ValueError(f"Benign calibration dataset is empty: {path}")
        group_ids = [example.question_id for example in examples]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError(
                "Benign calibration requires exactly one accepted rollout per independent prompt group"
            )
        return examples
