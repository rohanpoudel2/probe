from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import torch

from cli import run_output_confidence_baselines as confidence_runner
from data.generation_confidence import build_generation_confidence_trace
from data.schema import TaskExample


def _example(example_id: str, group: str, split: str, label: int) -> TaskExample:
    if label:
        token_ids = [2, 2]
        scores = [torch.tensor([0.0, 0.0, 0.0])] * 2
    else:
        token_ids = [0, 0]
        scores = [torch.tensor([4.0, 0.0, 0.0])] * 2
    generation = {
        "response_token_count": len(token_ids),
        "response_token_ids": token_ids,
        "confidence_trace": build_generation_confidence_trace(scores, token_ids),
    }
    return TaskExample(
        example_id=example_id,
        task_family="fixture",
        prompt=f"prompt {example_id}",
        label=label,
        question_id=group,
        assistant_response=f"response {example_id}",
        metadata={
            "protocol_split": split,
            "data_origin": "on_policy_generation",
            "generated_by_model": True,
            "model_id": "org/monitored",
            "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "generation": generation,
        },
    )


def test_output_confidence_runner_writes_metrics_and_predictions(
    tmp_path, monkeypatch
) -> None:
    source_path = tmp_path / "source.jsonl"
    target_path = tmp_path / "target.jsonl"
    calibration_path = tmp_path / "calibration.jsonl"
    for path in (source_path, target_path, calibration_path):
        path.write_text("fixture\n", encoding="utf-8")
    datasets = {
        str(source_path): [
            _example("train-pos", "train", "train", 1),
            _example("train-neg", "train", "train", 0),
            _example("eval-pos", "eval", "eval", 1),
            _example("eval-neg", "eval", "eval", 0),
            _example("test-pos", "test", "test", 1),
            _example("test-neg", "test", "test", 0),
        ],
        str(target_path): [
            _example("target-pos", "target", "test", 1),
            _example("target-neg", "target", "test", 0),
        ],
        str(calibration_path): [
            _example("cal-0", "cal-0", "calibration", 0),
            _example("cal-1", "cal-1", "calibration", 0),
        ],
    }

    class FixtureTask:
        spec = SimpleNamespace(grouped_split_key="question_id")

        @staticmethod
        def load(path):
            return datasets[path]

    monkeypatch.setattr(
        confidence_runner,
        "TASK_REGISTRY",
        {
            "source": FixtureTask,
            "target": FixtureTask,
            "benign_calibration": FixtureTask,
        },
    )
    results_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_output_confidence_baselines",
            "--source_task",
            "source",
            "--source_data",
            str(source_path),
            "--target_task",
            "target",
            "--target_data",
            str(target_path),
            "--calibration_task",
            "benign_calibration",
            "--calibration_data",
            str(calibration_path),
            "--model",
            "monitored",
            "--results_dir",
            str(results_dir),
            "--k_values",
            "1",
            "--seeds",
            "1",
            "--min_calibration_negatives",
            "2",
        ],
    )

    confidence_runner.main()

    output = results_dir / "source__to__target__output_confidence_baselines.jsonl"
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["probe"] == "B4_output_confidence_logistic"
    assert row["threshold_source"] == "source_calibration_negatives"
    assert row["transfer_recall_at_frozen_fpr"] == 1.0
    assert list((results_dir / "predictions").glob("*.jsonl"))
