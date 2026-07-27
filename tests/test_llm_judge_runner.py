from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np

from cli import run_llm_judge_baselines as judge_runner
from data.schema import TaskExample


def _example(example_id: str, group: str, split: str, label: int) -> TaskExample:
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
        },
    )


def test_llm_judge_runner_uses_resumable_cache_without_transformers(
    tmp_path, monkeypatch
) -> None:
    source_path = tmp_path / "source.jsonl"
    target_path = tmp_path / "target.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    judge_lock = tmp_path / "judge.yaml"
    for path in (source_path, target_path, reference_path, judge_lock):
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
        str(reference_path): [
            _example("cal-0", "cal-0", "calibration", 0),
            _example("cal-1", "cal-1", "calibration", 0),
            _example("holdout-0", "holdout-0", "test", 0),
            _example("holdout-1", "holdout-1", "test", 0),
        ],
    }

    class FixtureTask:
        spec = SimpleNamespace(grouped_split_key="question_id")

        @staticmethod
        def load(path):
            return datasets[path]

    class FakeJudgeRuntime:
        calls = 0

        def __init__(self, spec, *, batch_size, device="auto"):
            assert spec["model_id"] == "org/judge"
            assert batch_size == 2
            assert isinstance(device, str)
            self.batch_size = batch_size
            self.resolved_device = "cpu"
            self.model_parameter_dtype = "torch.float32"

        def score_bundle(self, bundle, demonstrations):
            self.__class__.calls += 1
            assert all(label in {0, 1} for _, label in demonstrations)
            return {
                "labels": bundle["labels"],
                "scores": 0.1 + 0.8 * bundle["labels"].astype(float),
                "example_ids": bundle["example_ids"],
                "question_ids": bundle["question_ids"],
                "prompt_sha256": np.asarray(
                    [
                        hashlib.sha256(text.encode("utf-8")).hexdigest()
                        for text in bundle["texts"]
                    ]
                ),
                "prompt_token_lengths": np.asarray(
                    [len(text) for text in bundle["texts"]], dtype=np.int64
                ),
            }

    spec = {
        "model_id": "org/judge",
        "model_revision": "c" * 40,
        "tokenizer_revision": "d" * 40,
        "family": "independent-family",
        "max_length": 1024,
        "negative_label": "A",
        "positive_label": "B",
        "protocol_role": "frozen_primary",
    }
    monkeypatch.setattr(
        judge_runner,
        "TASK_REGISTRY",
        {
            "source": FixtureTask,
            "target": FixtureTask,
            "reference_traffic": FixtureTask,
        },
    )
    monkeypatch.setattr(
        judge_runner, "_load_judge_spec", lambda path, key: (spec, "e" * 64, "f" * 64)
    )
    monkeypatch.setattr(judge_runner, "_git_state", lambda: ("1" * 40, False))
    monkeypatch.setattr(judge_runner, "_implementation_sha256", lambda: "2" * 64)
    monkeypatch.setattr(judge_runner, "JudgeRuntime", FakeJudgeRuntime)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_llm_judge_baselines",
            "--source_task",
            "source",
            "--source_data",
            str(source_path),
            "--target_task",
            "target",
            "--target_data",
            str(target_path),
            "--reference_task",
            "reference_traffic",
            "--reference_data",
            str(reference_path),
            "--judge_config",
            str(judge_lock),
            "--judge_model_key",
            "primary",
            "--judge_cache_dir",
            str(tmp_path / "cache"),
            "--judge_batch_size",
            "2",
            "--model",
            "monitored",
            "--results_dir",
            str(results_dir),
            "--views",
            "answer_text",
            "--modes",
            "zero_shot,few_shot",
            "--k_values",
            "1",
            "--seeds",
            "1",
            "--min_reference_groups",
            "2",
            "--max_reference_alert_rate",
            "0.5",
        ],
    )

    judge_runner.main()

    output = results_dir / "source__to__target__llm_judge_baselines.jsonl"
    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["probe"] for row in rows} == {
        "B3_llm_judge_zero_shot",
        "B3_llm_judge_few_shot",
    }
    assert all(
        row["transfer_tpr_at_reference_alert_budget"] == 1.0 for row in rows
    )
    assert all(row["reference_holdout_alert_rate"] == 0.0 for row in rows)
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2
    calls_after_first_run = FakeJudgeRuntime.calls

    monkeypatch.setattr(sys, "argv", [*sys.argv, "--overwrite"])
    judge_runner.main()
    assert FakeJudgeRuntime.calls == calls_after_first_run
