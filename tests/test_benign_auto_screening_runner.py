from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from cli import run_benign_auto_screening as runner
from data.benign_audit import build_automated_consensus


def _rollout(index: int) -> dict:
    prompt = [{"role": "user", "content": f"Explain topic {index}."}]
    response = f"A complete benign response about topic {index}."
    return {
        "rollout_id": f"rollout-{index}",
        "group_id": f"group-{index}",
        "task_family": "benign_calibration",
        "protocol_split": "calibration",
        "prompt_messages": prompt,
        "messages": [*prompt, {"role": "assistant", "content": response}],
        "response_text": response,
        "data_origin": "on_policy_generation",
        "generated_by_model": True,
        "model_id": "org/monitored",
        "generation": {
            "max_new_tokens": 256,
            "response_token_count": 12,
            "stop_reason": "eos_token",
        },
    }


def test_auto_screening_runner_is_resumable_and_consensus_ready(
    tmp_path, monkeypatch
) -> None:
    rollouts = [_rollout(0), _rollout(1)]
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rollouts),
        encoding="utf-8",
    )
    config_path = tmp_path / "screeners.yaml"
    config_path.write_text("fixture\n", encoding="utf-8")
    output_path = tmp_path / "decisions.jsonl"
    specs = {
        f"screener-{index}": {
            "protocol_role": "frozen_primary",
            "model_id": f"org/screener-{index}",
            "model_revision": str(index + 1) * 40,
            "tokenizer_revision": str(index + 4) * 40,
            "family": f"family-{index}",
            "eligible_max_probability": 0.10,
        }
        for index in range(3)
    }

    class FakeRuntime:
        calls = 0

        def __init__(self, spec, *, batch_size, device):
            self.spec = spec
            self.batch_size = batch_size
            self.device = device
            self.__class__.calls += 1

        def score_bundle(self, bundle, demonstrations):
            assert demonstrations == []
            return {
                "scores": np.full(len(bundle["texts"]), 0.01),
                "prompt_sha256": np.asarray(
                    [
                        hashlib.sha256(text.encode("utf-8")).hexdigest()
                        for text in bundle["texts"]
                    ]
                ),
                "prompt_token_lengths": np.asarray(
                    [len(text) for text in bundle["texts"]]
                ),
            }

    monkeypatch.setattr(
        runner,
        "_load_judge_spec",
        lambda path, key: (specs[key], "a" * 64, key[-1] * 64),
    )
    monkeypatch.setattr(runner, "_git_state", lambda: ("b" * 40, False))
    monkeypatch.setattr(runner, "_implementation_sha256", lambda: "c" * 64)
    monkeypatch.setattr(runner, "JudgeRuntime", FakeRuntime)
    monkeypatch.setattr(runner, "_release_runtime", lambda runtime: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benign_auto_screening",
            "--rollouts",
            str(rollout_path),
            "--screener_config",
            str(config_path),
            "--screener_model_keys",
            "screener-0,screener-1,screener-2",
            "--monitored_family",
            "Qwen",
            "--output",
            str(output_path),
            "--device",
            "cpu",
        ],
    )

    runner.main()
    calls_after_first_run = FakeRuntime.calls
    runner.main()
    assert FakeRuntime.calls == calls_after_first_run

    decisions = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(decisions) == 6
    consensus, report = build_automated_consensus(
        rollouts,
        decisions,
        monitored_family="Qwen",
    )
    assert report["status"] == "pass"
    assert all(row["decision"] == "eligible" for row in consensus)
