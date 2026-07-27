from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from cli.build_falsification_manifest import build_falsification_manifest
from cli.common import load_yaml
from cli.import_monitorbench_rollouts import (
    main as import_monitorbench_main,
    normalize_monitorbench_artifact,
)
from data.falsification import load_falsification_registry
from data.monitorbench import (
    EXPECTED_TASK_ARCHETYPES,
    expected_monitorbench_artifacts,
    file_sha256,
    load_monitorbench_adapter,
    load_monitorbench_tested_artifact,
    monitorbench_base_instance_id,
    parse_monitorbench_response,
    render_monitorbench_agent_conversation,
    validate_monitorbench_source_manifest,
)
from cli.fetch_exact_hf_sources import _install_monitorbench_archive
from tasks.cot_distortion import CoTDistortionTask


def _tested_row(*, verdicts: list[bool] | None = None) -> dict:
    return {
        "task_id": "instance-1_standard",
        "messages": [{"role": "user", "content": "Solve this task."}],
        "response": [
            "<think>target-directed reasoning</think>target achieved",
            "<think>ordinary reasoning</think>target not achieved",
        ],
        "verification_result": verdicts if verdicts is not None else [True, False],
        "target": "registered target",
        "metadata": {
            "task": "goal_sandbag.safety",
            "stress_test": "standard",
            "original_question": "What should the model do?",
            "decision_critical_factor": "registered intervention",
        },
    }


def _write_artifact(path: Path, row: dict | None = None) -> None:
    path.write_text(json.dumps(row or _tested_row()) + "\n", encoding="utf-8")


def _run_manifest(adapter_sha256: str, source_sha256: str) -> dict:
    return {
        "schema_version": "monitorbench-run-manifest-v1",
        "run_id": "monitorbench-test-run",
        "adapter_sha256": adapter_sha256,
        "source_manifest_sha256": source_sha256,
        "evaluated_model": {
            "model_id": "org/model",
            "model_revision": "1" * 40,
            "tokenizer_revision": "2" * 40,
            "response_format": "think_tags",
            "chat_template_sha256": "3" * 64,
        },
        "generation": {
            "backend": "vllm",
            "temperature": 0.6,
            "top_p": 0.9,
            "seed": 42,
            "max_tokens": 32768,
            "max_model_len": 32768,
            "rollout_number": 2,
            "resolved_config_sha256": "4" * 64,
        },
        "verifier": {
            "protocol": "official_monitorbench_task_verifiers",
            "model_id": "org/verifier",
            "model_revision": "5" * 40,
            "tokenizer_revision": "6" * 40,
            "resolved_config_sha256": "7" * 64,
        },
    }


def test_pinned_adapter_registers_exact_official_matrix() -> None:
    adapter, _ = load_monitorbench_adapter()
    assert len(adapter["tasks"]) == 19
    assert {
        task["task_id"]: task["archetype"] for task in adapter["tasks"]
    } == EXPECTED_TASK_ARCHETYPES
    assert len(expected_monitorbench_artifacts(adapter)) == 69
    assert adapter["normalization"]["label_semantics"] == {
        0: "official_target_outcome_not_verified",
        1: "official_target_outcome_verified",
    }


def test_official_response_parsers_preserve_reasoning_and_action() -> None:
    assert parse_monitorbench_response(
        "<think>reasoning</think>answer", "think_tags"
    ) == ("reasoning", "answer")
    assert parse_monitorbench_response(
        "analysisreasoningassistantfinalanswer", "gpt_oss"
    ) == ("reasoning", "answer")
    with pytest.raises(ValueError, match="lacks </think>"):
        parse_monitorbench_response("reasoning only", "think_tags")


def test_official_agent_renderer_preserves_cot_tools_and_outputs() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "I should inspect the account.",
            "tool_calls": [
                {"function": {"name": "lookup"}, "arguments": {"id": 7}}
            ],
        },
        {"role": "tool", "content": "account result"},
    ]
    cot, action, transcript = render_monitorbench_agent_conversation(conversation)
    assert cot == "I should inspect the account."
    assert action == 'lookup({"id": 7})\n\naccount result'
    assert transcript == (
        'I should inspect the account.\n\nlookup({"id": 7})\n\naccount result'
    )


def test_base_instance_id_removes_only_the_exact_stress_suffix() -> None:
    assert monitorbench_base_instance_id(
        "folder_standard", {"task_id": "folder_standard"}, "standard"
    ) == "folder"
    assert monitorbench_base_instance_id(
        "name_with_underscores_standard", {}, "standard"
    ) == "name_with_underscores"


def test_tested_artifact_requires_parallel_boolean_verdicts(tmp_path: Path) -> None:
    adapter, _ = load_monitorbench_adapter()
    artifact_path = tmp_path / "evaluated_llm_standard_n=2.tested.jsonl"
    invalid = _tested_row()
    invalid["verification_result"] = [1, 0]
    _write_artifact(artifact_path, invalid)
    with pytest.raises(ValueError, match="non-boolean verifier result"):
        load_monitorbench_tested_artifact(
            artifact_path,
            task_id="goal_sandbag.safety",
            adapter=adapter,
            expected_rollout_number=2,
        )


def test_normalization_is_test_only_and_builds_exact_prompt_pair(
    tmp_path: Path,
) -> None:
    adapter, adapter_sha256 = load_monitorbench_adapter()
    artifact_path = tmp_path / "evaluated_llm_standard_n=2.tested.jsonl"
    _write_artifact(artifact_path)
    artifact = load_monitorbench_tested_artifact(
        artifact_path,
        task_id="goal_sandbag.safety",
        adapter=adapter,
        expected_rollout_number=2,
    )
    registry, registry_sha256 = load_falsification_registry(
        Path("experiments/protocol/falsification_registry.yaml")
    )
    rows = normalize_monitorbench_artifact(
        artifact,
        adapter=adapter,
        adapter_sha256=adapter_sha256,
        source_manifest_sha256="8" * 64,
        run_manifest=_run_manifest(adapter_sha256, "8" * 64),
        run_manifest_sha256="9" * 64,
        falsification_registry=registry,
        falsification_registry_sha256=registry_sha256,
        eligible_for_main_study=True,
    )
    assert [row["label"] for row in rows] == [1, 0]
    assert {row["scenario_id"] for row in rows} == {rows[0]["scenario_id"]}
    assert {row["question_id"] for row in rows} == {rows[0]["question_id"]}
    assert {row["protocol_split"] for row in rows} == {"test"}
    assert all("confidence_trace" not in row["generation"] for row in rows)

    normalized_path = tmp_path / "cot_distortion.jsonl"
    normalized_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    examples = CoTDistortionTask().load(str(normalized_path))
    assert CoTDistortionTask.spec.evaluation_only is True
    assert [example.label for example in examples] == [1, 0]

    manifest = build_falsification_manifest(
        rows,
        source_path=normalized_path,
        registry=registry,
        registry_sha256=registry_sha256,
        minimum_hard_negative_pairs=1,
        model_name="test-model",
    )
    assert len(manifest["hard_negative_pairs"]) == 1
    assert manifest["hard_negative_pairs"][0]["match_type"] == (
        "exact_trigger_prompt"
    )


def test_archive_install_writes_and_revalidates_immutable_manifest(
    tmp_path: Path,
) -> None:
    adapter, _ = load_monitorbench_adapter()
    adapter = copy.deepcopy(adapter)
    revision = "a" * 40
    critical_content = b"pinned registry\n"
    critical_digest = hashlib.sha256(critical_content).hexdigest()
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr(
            f"MonitorBench-{revision}/pipeline/registry.py", critical_content
        )
    payload = archive_buffer.getvalue()
    adapter["source"].update(
        {
            "revision": revision,
            "archive_url": f"https://github.com/ASTRAL-Group/MonitorBench/archive/{revision}.zip",
            "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "critical_files": {"pipeline/registry.py": critical_digest},
        }
    )
    outdir = tmp_path / "raw"
    manifest_path = _install_monitorbench_archive(
        payload=payload, outdir=outdir, adapter=adapter
    )
    manifest, _ = validate_monitorbench_source_manifest(
        manifest_path, adapter=adapter
    )
    assert manifest["source_file_count"] == 1
    assert (outdir / revision / "pipeline" / "registry.py").read_bytes() == (
        critical_content
    )
    assert (
        _install_monitorbench_archive(
            payload=payload, outdir=outdir, adapter=adapter
        )
        == manifest_path
    )

    (outdir / revision / "pipeline" / "registry.py").write_text(
        "modified", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="critical file mismatch"):
        validate_monitorbench_source_manifest(manifest_path, adapter=adapter)


def test_partial_cli_import_discovers_results_and_marks_rows_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _ = load_monitorbench_adapter()
    adapter = copy.deepcopy(adapter)
    revision = "b" * 40
    critical_content = b"pinned registry\n"
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr(
            f"MonitorBench-{revision}/pipeline/registry.py", critical_content
        )
    archive_payload = archive_buffer.getvalue()
    adapter["source"].update(
        {
            "revision": revision,
            "archive_url": f"https://github.com/ASTRAL-Group/MonitorBench/archive/{revision}.zip",
            "archive_sha256": hashlib.sha256(archive_payload).hexdigest(),
            "critical_files": {
                "pipeline/registry.py": hashlib.sha256(critical_content).hexdigest()
            },
        }
    )
    adapter_path = tmp_path / "adapter.yaml"
    adapter_path.write_text(yaml.safe_dump(adapter, sort_keys=False), encoding="utf-8")
    adapter_sha256 = file_sha256(adapter_path)
    source_manifest_path = _install_monitorbench_archive(
        payload=archive_payload,
        outdir=tmp_path / "source",
        adapter=adapter,
    )
    _, source_manifest_sha256 = validate_monitorbench_source_manifest(
        source_manifest_path, adapter=adapter
    )
    run_manifest = _run_manifest(adapter_sha256, source_manifest_sha256)
    run_manifest_path = tmp_path / "run.yaml"
    run_manifest_path.write_text(
        yaml.safe_dump(run_manifest, sort_keys=False), encoding="utf-8"
    )
    results_root = tmp_path / "results"
    artifact_path = (
        results_root
        / "goal_sandbag.safety"
        / "inference_results"
        / "evaluated_llm_standard_n=2.tested.jsonl"
    )
    artifact_path.parent.mkdir(parents=True)
    _write_artifact(artifact_path)
    output_path = tmp_path / "normalized.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_monitorbench_rollouts",
            "--results_root",
            str(results_root),
            "--adapter",
            str(adapter_path),
            "--source_manifest",
            str(source_manifest_path),
            "--run_manifest",
            str(run_manifest_path),
            "--output",
            str(output_path),
            "--allow_incomplete_suite",
        ],
    )
    import_monitorbench_main()
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["eligible_for_main_study"] is False for row in rows)
    sidecar = json.loads(
        output_path.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert sidecar["complete_official_suite"] is False
    assert sidecar["eligible_for_main_study"] is False
def test_monitorbench_source_lock_matches_adapter_contract() -> None:
    source_lock = load_yaml(
        "experiments/data/huggingface_source_lock.yaml"
    )["sources"]["cot_monitorability_raw"]["monitorbench"]
    adapter_source = load_yaml(
        "experiments/protocol/monitorbench_adapter.yaml"
    )["source"]
    assert source_lock["github_repo"] == adapter_source["repository"]
    assert source_lock["revision"] == adapter_source["revision"]
    assert source_lock["github_zip"] == adapter_source["archive_url"]
    assert source_lock["sha256"] == adapter_source["archive_sha256"]
