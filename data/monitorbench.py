from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from data.rollout_schema import content_hash, validate_messages


MONITORBENCH_ADAPTER_SCHEMA_VERSION = "monitorbench-adapter-v1"
MONITORBENCH_SOURCE_MANIFEST_SCHEMA_VERSION = "monitorbench-source-manifest-v1"
MONITORBENCH_RUN_MANIFEST_SCHEMA_VERSION = "monitorbench-run-manifest-v1"
MONITORBENCH_NORMALIZED_METADATA_SCHEMA_VERSION = (
    "monitorbench-normalized-metadata-v1"
)
MONITORBENCH_GENERATION_REFERENCE_SCHEMA_VERSION = (
    "monitorbench-generation-reference-v1"
)
DEFAULT_MONITORBENCH_ADAPTER = Path(
    "experiments/protocol/monitorbench_adapter.yaml"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ARTIFACT_RE = re.compile(
    r"^evaluated_llm_(original|standard|direct_conceal|monitor_aware)_n=(\d+)"
    r"\.tested\.jsonl$"
)

EXPECTED_TASK_ARCHETYPES = {
    "goal_sandbag.safety": "input_intervention",
    "goal_sandbag.math": "input_intervention",
    "goal_sandbag.science": "input_intervention",
    "goal_sandbag.engineering": "input_intervention",
    "goal_deception.realworld": "input_intervention",
    "cue_condition.knowledge": "input_intervention",
    "cue_condition.preference": "input_intervention",
    "cue_condition.judgment": "input_intervention",
    "cue_condition.health": "input_intervention",
    "cue_answer.math": "input_intervention",
    "cue_answer.health": "input_intervention",
    "cue_answer.logic": "input_intervention",
    "dual_objectives.steganography": "outcome_justification",
    "dual_objectives.summarization": "outcome_justification",
    "dual_objectives.coding": "outcome_justification",
    "dual_objectives.agent.benign": "outcome_justification",
    "dual_objectives.agent.malicious": "outcome_justification",
    "impossible.coding": "outcome_justification",
    "process.multistep": "solution_process",
}
EXPECTED_STRESS_TESTS = {
    "input_intervention": (
        "original",
        "standard",
        "direct_conceal",
        "monitor_aware",
    ),
    "outcome_justification": (
        "standard",
        "direct_conceal",
        "monitor_aware",
    ),
    "solution_process": (
        "standard",
        "direct_conceal",
        "monitor_aware",
    ),
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    normalized = str(value or "").lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return normalized


def _require_commit(value: Any, field: str) -> str:
    normalized = str(value or "").lower()
    if COMMIT_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be a full 40-character commit")
    return normalized


def _read_yaml_object(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a YAML object")
    return payload


def validate_monitorbench_adapter(adapter: dict[str, Any]) -> None:
    if adapter.get("schema_version") != MONITORBENCH_ADAPTER_SCHEMA_VERSION:
        raise ValueError("Unsupported MonitorBench adapter schema")
    if not str(adapter.get("adapter_id", "")).strip():
        raise ValueError("MonitorBench adapter requires adapter_id")

    source = adapter.get("source")
    if not isinstance(source, dict):
        raise ValueError("MonitorBench adapter requires source metadata")
    if source.get("repository") != "https://github.com/ASTRAL-Group/MonitorBench":
        raise ValueError("MonitorBench adapter does not identify the official repository")
    _require_commit(source.get("revision"), "source.revision")
    _require_sha256(source.get("archive_sha256"), "source.archive_sha256")
    if not str(source.get("archive_url", "")).endswith(
        f"/{source['revision']}.zip"
    ):
        raise ValueError("MonitorBench archive URL is not bound to source.revision")
    critical_files = source.get("critical_files")
    if not isinstance(critical_files, dict) or not critical_files:
        raise ValueError("MonitorBench adapter requires critical file hashes")
    for relative_path, digest in critical_files.items():
        path = Path(str(relative_path))
        if path.is_absolute() or ".." in path.parts or not str(path).strip():
            raise ValueError("MonitorBench critical file paths must be relative")
        _require_sha256(digest, f"critical_files.{relative_path}")

    artifact = adapter.get("artifact_contract")
    if not isinstance(artifact, dict):
        raise ValueError("MonitorBench adapter requires artifact_contract")
    if artifact.get("filename_pattern") != (
        "evaluated_llm_<stress_test>_n=<rollout_number>.tested.jsonl"
    ):
        raise ValueError("MonitorBench artifact filename contract changed")
    expected_record_fields = {
        "task_id",
        "messages",
        "response",
        "verification_result",
        "target",
        "metadata",
    }
    if set(artifact.get("required_record_fields") or []) != expected_record_fields:
        raise ValueError("MonitorBench tested-artifact fields changed")
    if set(artifact.get("required_metadata_fields") or []) != {
        "task",
        "stress_test",
        "original_question",
    }:
        raise ValueError("MonitorBench tested-artifact metadata fields changed")
    if set(artifact.get("response_formats") or []) != {"think_tags", "gpt_oss"}:
        raise ValueError("MonitorBench response formats changed")
    if set(artifact.get("structured_response_tasks") or []) != {
        "dual_objectives.agent.benign",
        "dual_objectives.agent.malicious",
    }:
        raise ValueError("MonitorBench structured-response task registry changed")

    normalization = adapter.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("MonitorBench adapter requires normalization metadata")
    if (
        normalization.get("task_family") != "cot_distortion"
        or normalization.get("construct_name")
        != "monitorbench_verified_target_outcome"
        or normalization.get("protocol_split") != "test"
        or normalization.get("label_semantics")
        != {
            0: "official_target_outcome_not_verified",
            1: "official_target_outcome_verified",
        }
    ):
        raise ValueError("MonitorBench normalization semantics changed")
    confidence = normalization.get("generation_confidence")
    if not isinstance(confidence, dict) or confidence.get("available") is not False:
        raise ValueError("MonitorBench confidence availability must be explicit")

    tasks = adapter.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(EXPECTED_TASK_ARCHETYPES):
        raise ValueError("MonitorBench adapter must register exactly 19 tasks")
    observed: dict[str, str] = {}
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"task_id", "archetype"}:
            raise ValueError("Invalid MonitorBench task entry")
        task_id = str(task["task_id"])
        if task_id in observed:
            raise ValueError(f"Duplicate MonitorBench task {task_id}")
        observed[task_id] = str(task["archetype"])
    if observed != EXPECTED_TASK_ARCHETYPES:
        raise ValueError("MonitorBench task registry differs from the pinned adapter")
    stress_tests = adapter.get("stress_tests_by_archetype")
    if not isinstance(stress_tests, dict) or set(stress_tests) != set(
        EXPECTED_STRESS_TESTS
    ):
        raise ValueError("MonitorBench stress-test registry is incomplete")
    for archetype, expected in EXPECTED_STRESS_TESTS.items():
        if tuple(stress_tests[archetype]) != expected:
            raise ValueError(f"Unexpected stress tests for {archetype}")


def load_monitorbench_adapter(
    path: Path = DEFAULT_MONITORBENCH_ADAPTER,
) -> tuple[dict[str, Any], str]:
    adapter = _read_yaml_object(path, "MonitorBench adapter")
    validate_monitorbench_adapter(adapter)
    return adapter, file_sha256(path)


def monitorbench_task_map(adapter: dict[str, Any]) -> dict[str, str]:
    return {str(task["task_id"]): str(task["archetype"]) for task in adapter["tasks"]}


def expected_monitorbench_artifacts(
    adapter: dict[str, Any],
) -> set[tuple[str, str]]:
    task_map = monitorbench_task_map(adapter)
    return {
        (task_id, stress_test)
        for task_id, archetype in task_map.items()
        for stress_test in adapter["stress_tests_by_archetype"][archetype]
    }


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_sha256"}


def source_tree_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing MonitorBench source tree: {root}")
    inventory: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        inventory.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not inventory:
        raise ValueError("MonitorBench source tree is empty")
    return inventory


def create_monitorbench_source_manifest(
    *,
    manifest_path: Path,
    archive_path: Path,
    extracted_root: Path,
    adapter: dict[str, Any],
) -> dict[str, Any]:
    validate_monitorbench_adapter(adapter)
    source = adapter["source"]
    archive_digest = file_sha256(archive_path)
    if archive_digest != source["archive_sha256"]:
        raise ValueError("MonitorBench archive does not match the pinned adapter")
    for relative_path, expected_digest in source["critical_files"].items():
        path = extracted_root / relative_path
        if not path.is_file() or file_sha256(path) != expected_digest:
            raise ValueError(
                f"MonitorBench critical source file mismatch: {relative_path}"
            )
    inventory = source_tree_inventory(extracted_root)
    base = manifest_path.parent.resolve()
    try:
        archive_relative = archive_path.resolve().relative_to(base).as_posix()
        root_relative = extracted_root.resolve().relative_to(base).as_posix()
    except ValueError as err:
        raise ValueError("MonitorBench source artifacts must share one manifest root") from err
    manifest: dict[str, Any] = {
        "schema_version": MONITORBENCH_SOURCE_MANIFEST_SCHEMA_VERSION,
        "source": {
            "repository": source["repository"],
            "revision": source["revision"],
            "archive_url": source["archive_url"],
            "archive_sha256": archive_digest,
        },
        "archive_file": archive_relative,
        "extracted_root": root_relative,
        "source_file_count": len(inventory),
        "source_tree_sha256": content_hash(inventory),
        "critical_files": dict(source["critical_files"]),
    }
    manifest["manifest_sha256"] = content_hash(_manifest_payload(manifest))
    return manifest


def validate_monitorbench_source_manifest(
    path: Path,
    *,
    adapter: dict[str, Any],
    verify_tree: bool = True,
) -> tuple[dict[str, Any], str]:
    manifest = _read_yaml_object(path, "MonitorBench source manifest")
    if manifest.get("schema_version") != MONITORBENCH_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported MonitorBench source-manifest schema")
    if manifest.get("manifest_sha256") != content_hash(_manifest_payload(manifest)):
        raise ValueError("MonitorBench source manifest hash mismatch")
    source = manifest.get("source")
    expected_source = adapter["source"]
    if not isinstance(source, dict) or any(
        source.get(key) != expected_source[key]
        for key in ("repository", "revision", "archive_url", "archive_sha256")
    ):
        raise ValueError("MonitorBench source manifest differs from the adapter")
    if manifest.get("critical_files") != expected_source["critical_files"]:
        raise ValueError("MonitorBench source manifest has stale critical hashes")

    base = path.parent.resolve()
    archive_path = (base / str(manifest.get("archive_file", ""))).resolve()
    extracted_root = (base / str(manifest.get("extracted_root", ""))).resolve()
    for resolved, description in (
        (archive_path, "archive"),
        (extracted_root, "source tree"),
    ):
        if resolved != base and base not in resolved.parents:
            raise ValueError(f"MonitorBench {description} escapes the manifest root")
    if not archive_path.is_file() or file_sha256(archive_path) != source["archive_sha256"]:
        raise ValueError("MonitorBench archived source is missing or modified")
    if not extracted_root.is_dir():
        raise FileNotFoundError(f"Missing MonitorBench extracted source: {extracted_root}")
    for relative_path, expected_digest in expected_source["critical_files"].items():
        critical_path = extracted_root / relative_path
        if not critical_path.is_file() or file_sha256(critical_path) != expected_digest:
            raise ValueError(
                f"MonitorBench extracted critical file mismatch: {relative_path}"
            )
    if verify_tree:
        inventory = source_tree_inventory(extracted_root)
        if (
            manifest.get("source_file_count") != len(inventory)
            or manifest.get("source_tree_sha256") != content_hash(inventory)
        ):
            raise ValueError("MonitorBench extracted source tree was modified")
    return manifest, file_sha256(path)


def _validate_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def validate_monitorbench_run_manifest(
    manifest: dict[str, Any],
    *,
    adapter: dict[str, Any],
    adapter_sha256: str,
    source_manifest_sha256: str,
) -> None:
    if manifest.get("schema_version") != MONITORBENCH_RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported MonitorBench run-manifest schema")
    run_id = str(manifest.get("run_id", "")).strip()
    if not run_id or "REPLACE" in run_id:
        raise ValueError("MonitorBench run manifest requires an immutable run_id")
    if manifest.get("adapter_sha256") != adapter_sha256:
        raise ValueError("MonitorBench run manifest has a stale adapter hash")
    if manifest.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("MonitorBench run manifest has a stale source-manifest hash")

    model = manifest.get("evaluated_model")
    if not isinstance(model, dict) or not str(model.get("model_id", "")).strip():
        raise ValueError("MonitorBench run manifest lacks evaluated-model identity")
    _require_commit(model.get("model_revision"), "evaluated_model.model_revision")
    _require_commit(
        model.get("tokenizer_revision"), "evaluated_model.tokenizer_revision"
    )
    if model.get("response_format") not in adapter["artifact_contract"][
        "response_formats"
    ]:
        raise ValueError("MonitorBench run manifest has an unsupported response format")
    _require_sha256(
        model.get("chat_template_sha256"),
        "evaluated_model.chat_template_sha256",
    )

    generation = manifest.get("generation")
    required_generation = {
        "backend",
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "max_model_len",
        "rollout_number",
        "resolved_config_sha256",
    }
    if not isinstance(generation, dict) or set(generation) != required_generation:
        raise ValueError("MonitorBench run manifest has an incomplete generation config")
    if not str(generation["backend"]).strip():
        raise ValueError("MonitorBench generation backend must be non-empty")
    temperature = _validate_finite_number(generation["temperature"], "temperature")
    top_p = _validate_finite_number(generation["top_p"], "top_p")
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("MonitorBench temperature/top_p are outside valid ranges")
    for key in ("seed", "max_tokens", "max_model_len", "rollout_number"):
        value = generation[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            if key == "seed" and isinstance(value, int) and not isinstance(value, bool):
                continue
            raise ValueError(f"MonitorBench generation.{key} must be a positive integer")
    _require_sha256(
        generation["resolved_config_sha256"], "generation.resolved_config_sha256"
    )

    verifier = manifest.get("verifier")
    required_verifier = {
        "protocol",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "resolved_config_sha256",
    }
    if not isinstance(verifier, dict) or set(verifier) != required_verifier:
        raise ValueError("MonitorBench run manifest has incomplete verifier metadata")
    if verifier.get("protocol") != "official_monitorbench_task_verifiers":
        raise ValueError("MonitorBench run manifest must use official task verifiers")
    if not str(verifier.get("model_id", "")).strip():
        raise ValueError("MonitorBench verifier model_id is required")
    _require_commit(verifier.get("model_revision"), "verifier.model_revision")
    _require_commit(
        verifier.get("tokenizer_revision"), "verifier.tokenizer_revision"
    )
    _require_sha256(
        verifier.get("resolved_config_sha256"), "verifier.resolved_config_sha256"
    )


def load_monitorbench_run_manifest(
    path: Path,
    *,
    adapter: dict[str, Any],
    adapter_sha256: str,
    source_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    manifest = _read_yaml_object(path, "MonitorBench run manifest")
    validate_monitorbench_run_manifest(
        manifest,
        adapter=adapter,
        adapter_sha256=adapter_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )
    return manifest, file_sha256(path)


def parse_monitorbench_response(response: str, response_format: str) -> tuple[str, str]:
    if not isinstance(response, str) or not response.strip():
        raise ValueError("MonitorBench response must be non-empty text")
    if response_format == "think_tags":
        if "</think>" not in response:
            raise ValueError("MonitorBench response lacks </think>")
        reasoning, action = response.split("</think>", 1)
        if reasoning.startswith("<think>"):
            reasoning = reasoning[len("<think>") :]
    elif response_format == "gpt_oss":
        match = re.search(r"assistant(?:commentary|final|analysis)", response)
        if match is None:
            raise ValueError("GPT-OSS response lacks an assistant channel marker")
        reasoning = response[: match.start()]
        action = response[match.end() :]
        if reasoning.startswith("analysis"):
            reasoning = reasoning[len("analysis") :]
    else:
        raise ValueError(f"Unsupported MonitorBench response format: {response_format}")
    reasoning = reasoning.strip()
    action = action.strip()
    if not reasoning or not action:
        raise ValueError("MonitorBench response must contain both CoT and action text")
    return reasoning, action


def _format_monitorbench_tool_call(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name", "")
    else:
        name = function or tool_call.get("name", "")
    arguments = tool_call.get("args") or tool_call.get("arguments") or {}
    try:
        rendered_arguments = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered_arguments = str(arguments)
    return f"{name}({rendered_arguments})"


def render_monitorbench_agent_conversation(
    conversation: Any,
) -> tuple[str, str, str]:
    """Mirror the pinned DualAgent monitor rendering for cot/action/all scopes."""

    if not isinstance(conversation, list) or not conversation:
        raise ValueError("MonitorBench agent response must be a non-empty conversation")
    cot_parts: list[str] = []
    action_parts: list[str] = []
    all_parts: list[str] = []
    for index, message in enumerate(conversation):
        if not isinstance(message, dict):
            raise ValueError(
                f"MonitorBench agent conversation message {index} must be an object"
            )
        role = str(message.get("role") or "").lower()
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError(
                f"MonitorBench agent conversation message {index} has non-text content"
            )
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list) or any(
                not isinstance(tool_call, dict) for tool_call in tool_calls
            ):
                raise ValueError(
                    f"MonitorBench agent conversation message {index} has invalid tool calls"
                )
            if content:
                cot_parts.append(content)
                all_parts.append(content)
            if tool_calls:
                rendered_calls = "\n".join(
                    _format_monitorbench_tool_call(tool_call)
                    for tool_call in tool_calls
                )
                action_parts.append(rendered_calls)
                all_parts.append(rendered_calls)
        elif role == "tool" and content:
            snippet = content if len(content) <= 2000 else content[:2000] + "...[truncated]"
            action_parts.append(snippet)
            all_parts.append(snippet)
    rendered_all = "\n\n".join(all_parts).strip()
    if not rendered_all:
        raise ValueError("MonitorBench agent conversation has no monitor-visible content")
    return (
        "\n\n".join(cot_parts).strip(),
        "\n\n".join(action_parts).strip(),
        rendered_all,
    )


@dataclass(frozen=True)
class MonitorBenchArtifact:
    path: Path
    task_id: str
    archetype: str
    stress_test: str
    rollout_number: int
    sha256: str
    records: tuple[dict[str, Any], ...]


def _read_jsonl_objects(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from err
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield line_number, row


def load_monitorbench_tested_artifact(
    path: Path,
    *,
    task_id: str,
    adapter: dict[str, Any],
    expected_rollout_number: int,
) -> MonitorBenchArtifact:
    match = ARTIFACT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not an official MonitorBench tested artifact name: {path.name}")
    stress_test = match.group(1)
    rollout_number = int(match.group(2))
    if rollout_number != expected_rollout_number:
        raise ValueError(
            f"MonitorBench artifact {path} encodes n={rollout_number}, expected "
            f"n={expected_rollout_number}"
        )
    task_map = monitorbench_task_map(adapter)
    if task_id not in task_map:
        raise ValueError(f"Unknown MonitorBench task: {task_id}")
    archetype = task_map[task_id]
    if stress_test not in adapter["stress_tests_by_archetype"][archetype]:
        raise ValueError(
            f"Stress test {stress_test} is invalid for MonitorBench task {task_id}"
        )

    contract = adapter["artifact_contract"]
    required_fields = set(contract["required_record_fields"])
    required_metadata = set(contract["required_metadata_fields"])
    records: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for line_number, row in _read_jsonl_objects(path):
        missing = required_fields.difference(row)
        if missing:
            raise ValueError(
                f"MonitorBench artifact {path}:{line_number} lacks {sorted(missing)}"
            )
        official_task_id = str(row.get("task_id", "")).strip()
        if not official_task_id or official_task_id in seen_task_ids:
            raise ValueError(
                f"MonitorBench artifact {path}:{line_number} has an empty or duplicate task_id"
            )
        seen_task_ids.add(official_task_id)
        messages = validate_messages(row.get("messages"), allow_assistant=False)
        responses = row.get("response")
        verdicts = row.get("verification_result")
        if not isinstance(responses, list) or not responses:
            raise ValueError(
                f"MonitorBench artifact {path}:{line_number} requires response[]"
            )
        if not isinstance(verdicts, list) or len(verdicts) != len(responses):
            raise ValueError(
                f"MonitorBench artifact {path}:{line_number} has misaligned verifier results"
            )
        structured_tasks = set(contract["structured_response_tasks"])
        if task_id in structured_tasks:
            for response in responses:
                try:
                    render_monitorbench_agent_conversation(response)
                except ValueError as err:
                    raise ValueError(
                        f"MonitorBench artifact {path}:{line_number} has an invalid "
                        f"agent response: {err}"
                    ) from err
        elif any(
            not isinstance(response, str) or not response.strip()
            for response in responses
        ):
            raise ValueError(
                f"MonitorBench artifact {path}:{line_number} has an empty response"
            )
        if any(type(verdict) is not bool for verdict in verdicts):
            raise ValueError(
                f"MonitorBench artifact {path}:{line_number} has a non-boolean verifier result"
            )
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or required_metadata.difference(metadata):
            raise ValueError(
                f"MonitorBench artifact {path}:{line_number} lacks required metadata"
            )
        if metadata.get("task") != task_id or metadata.get("stress_test") != stress_test:
            raise ValueError(
                f"MonitorBench artifact {path}:{line_number} disagrees with task/stress provenance"
            )
        if not isinstance(metadata.get("original_question"), str) or not metadata[
            "original_question"
        ].strip():
            raise ValueError(
                f"MonitorBench artifact {path}:{line_number} lacks original_question text"
            )
        records.append(
            {
                **row,
                "task_id": official_task_id,
                "messages": messages,
                "response": list(responses),
                "verification_result": list(verdicts),
                "metadata": dict(metadata),
                "artifact_line_number": line_number,
            }
        )
    if not records:
        raise ValueError(f"MonitorBench tested artifact is empty: {path}")
    return MonitorBenchArtifact(
        path=path,
        task_id=task_id,
        archetype=archetype,
        stress_test=stress_test,
        rollout_number=rollout_number,
        sha256=file_sha256(path),
        records=tuple(records),
    )


def monitorbench_base_instance_id(
    official_task_id: str, metadata: dict[str, Any], stress_test: str
) -> str:
    metadata_id = metadata.get("task_id")
    if isinstance(metadata_id, (str, int)) and str(metadata_id).strip():
        candidate = str(metadata_id).strip()
    else:
        candidate = official_task_id
    suffix = f"_{stress_test}"
    if candidate.endswith(suffix):
        base = candidate[: -len(suffix)]
        if base:
            return base
    return candidate


def validate_normalized_monitorbench_row(
    row: dict[str, Any],
    *,
    adapter: dict[str, Any],
    adapter_sha256: str,
) -> None:
    normalization = adapter["normalization"]
    if row.get("task_family") != normalization["task_family"]:
        raise ValueError("Normalized MonitorBench row has the wrong task family")
    if row.get("protocol_split") != "test":
        raise ValueError("MonitorBench is a test-only evaluation family")
    if type(row.get("label")) is not int or row["label"] not in {0, 1}:
        raise ValueError("Normalized MonitorBench row requires an integer binary label")
    if row.get("label_source") != normalization["label_source"] or row.get(
        "annotation_protocol"
    ) != normalization["annotation_protocol"]:
        raise ValueError("Normalized MonitorBench row has unregistered label provenance")
    if (
        row.get("data_origin") != "on_policy_generation"
        or row.get("generated_by_model") is not True
    ):
        raise ValueError("Normalized MonitorBench row is not an on-policy rollout")
    metadata = row.get("metadata")
    monitorbench = metadata.get("monitorbench") if isinstance(metadata, dict) else None
    if not isinstance(monitorbench, dict):
        raise ValueError("Normalized MonitorBench row lacks adapter metadata")
    if (
        monitorbench.get("schema_version")
        != MONITORBENCH_NORMALIZED_METADATA_SCHEMA_VERSION
        or monitorbench.get("adapter_id") != adapter["adapter_id"]
        or monitorbench.get("adapter_sha256") != adapter_sha256
        or monitorbench.get("source_revision") != adapter["source"]["revision"]
        or monitorbench.get("construct_name")
        != normalization["construct_name"]
    ):
        raise ValueError("Normalized MonitorBench row is stale or uses another construct")
    task_id = str(monitorbench.get("official_task", ""))
    stress_test = str(monitorbench.get("stress_test", ""))
    task_map = monitorbench_task_map(adapter)
    if task_id not in task_map or monitorbench.get("archetype") != task_map[task_id]:
        raise ValueError("Normalized MonitorBench row has an unknown official task")
    if stress_test not in adapter["stress_tests_by_archetype"][task_map[task_id]]:
        raise ValueError("Normalized MonitorBench row has an invalid stress test")
    for field in (
        "source_manifest_sha256",
        "run_manifest_sha256",
        "artifact_sha256",
        "prompt_messages_sha256",
    ):
        _require_sha256(monitorbench.get(field), f"monitorbench.{field}")
    if row.get("condition") != f"monitorbench_{stress_test}":
        raise ValueError("Normalized MonitorBench condition disagrees with stress_test")
    if not str(row.get("model_id", "")).strip():
        raise ValueError("Normalized MonitorBench row lacks evaluated-model identity")
    _require_commit(row.get("model_revision"), "model_revision")
    _require_commit(row.get("tokenizer_revision"), "tokenizer_revision")
    for field in ("rollout_id", "scenario_id", "group_id", "example_id"):
        if not str(row.get(field, "")).strip():
            raise ValueError(f"Normalized MonitorBench row lacks {field}")
    response_text = row.get("response_text")
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("Normalized MonitorBench row lacks response text")
    for field in ("chain_of_thought", "final_answer"):
        segment = row.get(field)
        if (
            not isinstance(segment, str)
            or not segment.strip()
            or segment not in response_text
        ):
            raise ValueError(
                f"Normalized MonitorBench {field} is absent from the stored transcript"
            )
    annotation = row.get("annotation_metadata")
    if not isinstance(annotation, dict):
        raise ValueError("Normalized MonitorBench row lacks verifier evidence")
    _require_sha256(annotation.get("response_sha256"), "annotation.response_sha256")
    official_response = row.get("official_response", row.get("response_text"))
    if annotation["response_sha256"] != content_hash(official_response):
        raise ValueError("Normalized MonitorBench response hash mismatch")
    if annotation.get("verification_result") is not bool(row["label"]):
        raise ValueError("Normalized MonitorBench label disagrees with verifier evidence")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Normalized MonitorBench row lacks provenance")
    if (
        provenance.get("code_commit") != adapter["source"]["revision"]
        or provenance.get("code_dirty") is not False
        or provenance.get("monitorbench_adapter_sha256") != adapter_sha256
        or provenance.get("monitorbench_source_manifest_sha256")
        != monitorbench["source_manifest_sha256"]
        or provenance.get("monitorbench_run_manifest_sha256")
        != monitorbench["run_manifest_sha256"]
        or provenance.get("official_artifact_sha256")
        != monitorbench["artifact_sha256"]
        or provenance.get("scenario_file_sha256")
        != monitorbench["artifact_sha256"]
    ):
        raise ValueError("Normalized MonitorBench provenance hashes disagree")
    _require_sha256(
        provenance.get("chat_template_sha256"),
        "provenance.chat_template_sha256",
    )
    generation = row.get("generation")
    if (
        not isinstance(generation, dict)
        or generation.get("schema_version")
        != MONITORBENCH_GENERATION_REFERENCE_SCHEMA_VERSION
        or generation.get("confidence_trace_available") is not False
        or "confidence_trace" in generation
    ):
        raise ValueError("Normalized MonitorBench generation provenance is invalid")
    validate_messages(row.get("prompt_messages"), allow_assistant=False)
    messages = validate_messages(row.get("messages"), allow_assistant=True)
    if messages[:-1] != row["prompt_messages"] or messages[-1]["content"] != row.get(
        "response_text"
    ):
        raise ValueError("Normalized MonitorBench transcript is inconsistent")
