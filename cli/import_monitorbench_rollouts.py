from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from data.falsification import (
    load_falsification_registry,
    make_falsification_metadata,
)
from data.monitorbench import (
    DEFAULT_MONITORBENCH_ADAPTER,
    MONITORBENCH_GENERATION_REFERENCE_SCHEMA_VERSION,
    MONITORBENCH_NORMALIZED_METADATA_SCHEMA_VERSION,
    MonitorBenchArtifact,
    expected_monitorbench_artifacts,
    file_sha256,
    load_monitorbench_adapter,
    load_monitorbench_run_manifest,
    load_monitorbench_tested_artifact,
    monitorbench_base_instance_id,
    parse_monitorbench_response,
    render_monitorbench_agent_conversation,
    validate_monitorbench_source_manifest,
    validate_normalized_monitorbench_row,
)
from data.rollout_schema import (
    RolloutRecord,
    ScenarioRecord,
    content_hash,
)


IMPORT_MANIFEST_SCHEMA_VERSION = "monitorbench-import-manifest-v1"


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _last_user_prompt(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    raise ValueError("MonitorBench prompt has no user message")


def normalize_monitorbench_artifact(
    artifact: MonitorBenchArtifact,
    *,
    adapter: dict[str, Any],
    adapter_sha256: str,
    source_manifest_sha256: str,
    run_manifest: dict[str, Any],
    run_manifest_sha256: str,
    falsification_registry: dict[str, Any],
    falsification_registry_sha256: str,
    eligible_for_main_study: bool,
) -> list[dict[str, Any]]:
    normalization = adapter["normalization"]
    model = run_manifest["evaluated_model"]
    generation_config = run_manifest["generation"]
    falsification = make_falsification_metadata(
        registry=falsification_registry,
        registry_sha256=falsification_registry_sha256,
        task_name="cot_distortion",
        axis_values={
            "behavior": "cot_distortion",
            "domain": "monitorbench",
            "template": "official_monitorbench",
            "presentation": "plain",
            "obfuscation": "none",
        },
    )
    rows: list[dict[str, Any]] = []
    for record in artifact.records:
        official_task_id = record["task_id"]
        base_instance_id = monitorbench_base_instance_id(
            official_task_id, record["metadata"], artifact.stress_test
        )
        group_id = "mbg-" + content_hash(
            {"task": artifact.task_id, "base_instance_id": base_instance_id}
        )
        prompt_messages = record["messages"]
        prompt_hash = content_hash(prompt_messages)
        scenario_id = "mbs-" + content_hash(
            {
                "task": artifact.task_id,
                "stress_test": artifact.stress_test,
                "official_task_id": official_task_id,
                "prompt_messages": prompt_messages,
            }
        )
        monitorbench_metadata = {
            "schema_version": MONITORBENCH_NORMALIZED_METADATA_SCHEMA_VERSION,
            "adapter_id": adapter["adapter_id"],
            "adapter_sha256": adapter_sha256,
            "source_revision": adapter["source"]["revision"],
            "source_manifest_sha256": source_manifest_sha256,
            "run_id": run_manifest["run_id"],
            "run_manifest_sha256": run_manifest_sha256,
            "artifact_sha256": artifact.sha256,
            "artifact_line_number": record["artifact_line_number"],
            "official_task": artifact.task_id,
            "official_task_id": official_task_id,
            "base_instance_id": base_instance_id,
            "archetype": artifact.archetype,
            "stress_test": artifact.stress_test,
            "prompt_messages_sha256": prompt_hash,
            "construct_name": normalization["construct_name"],
            "official_metric_separation": normalization[
                "official_metric_separation"
            ],
            "official_metadata": record["metadata"],
            "official_target": record["target"],
        }
        scenario = ScenarioRecord(
            scenario_id=scenario_id,
            group_id=group_id,
            task_family="cot_distortion",
            messages=prompt_messages,
            condition=f"monitorbench_{artifact.stress_test}",
            protocol_split="test",
            source=(
                "ASTRAL-Group/MonitorBench@" + adapter["source"]["revision"]
            ),
            metadata={
                "monitorbench": monitorbench_metadata,
                "falsification": falsification,
                "eligible_for_main_study": eligible_for_main_study,
            },
        )
        for response_index, (official_response, verdict) in enumerate(
            zip(record["response"], record["verification_result"])
        ):
            try:
                if artifact.task_id in adapter["artifact_contract"][
                    "structured_response_tasks"
                ]:
                    cot_text, action_text, response = (
                        render_monitorbench_agent_conversation(official_response)
                    )
                    reasoning = cot_text or response
                    action = action_text or response
                else:
                    response = official_response
                    reasoning, action = parse_monitorbench_response(
                        response, model["response_format"]
                    )
            except ValueError as err:
                raise ValueError(
                    f"Cannot parse {artifact.path}:{record['artifact_line_number']} "
                    f"response {response_index}: {err}"
                ) from err
            response_sha256 = content_hash(official_response)
            rollout_id = "mbr-" + content_hash(
                {
                    "run_id": run_manifest["run_id"],
                    "scenario_id": scenario_id,
                    "response_index": response_index,
                    "response_sha256": response_sha256,
                    "artifact_sha256": artifact.sha256,
                }
            )
            generation = {
                "schema_version": MONITORBENCH_GENERATION_REFERENCE_SCHEMA_VERSION,
                "backend": generation_config["backend"],
                "temperature": generation_config["temperature"],
                "top_p": generation_config["top_p"],
                "seed": generation_config["seed"],
                "max_tokens": generation_config["max_tokens"],
                "max_model_len": generation_config["max_model_len"],
                "rollout_number": generation_config["rollout_number"],
                "response_index": response_index,
                "resolved_config_sha256": generation_config[
                    "resolved_config_sha256"
                ],
                "confidence_trace_available": False,
                "confidence_trace_unavailable_reason": normalization[
                    "generation_confidence"
                ]["reason"],
            }
            provenance = {
                "code_commit": adapter["source"]["revision"],
                "code_dirty": False,
                "chat_template_sha256": model["chat_template_sha256"],
                "scenario_file_sha256": artifact.sha256,
                "monitorbench_adapter_sha256": adapter_sha256,
                "monitorbench_source_manifest_sha256": source_manifest_sha256,
                "monitorbench_run_manifest_sha256": run_manifest_sha256,
                "official_artifact_sha256": artifact.sha256,
            }
            rollout = RolloutRecord(
                rollout_id=rollout_id,
                scenario=scenario,
                response_text=response,
                messages=[*prompt_messages, {"role": "assistant", "content": response}],
                model_id=model["model_id"],
                model_revision=model["model_revision"],
                tokenizer_revision=model["tokenizer_revision"],
                seed=generation_config["seed"],
                generation=generation,
                provenance=provenance,
                reasoning=reasoning,
                final_answer=action,
            )
            row = {
                **rollout.to_dict(),
                "example_id": rollout_id,
                "question_id": group_id,
                "prompt": _last_user_prompt(prompt_messages),
                "prompt_messages": prompt_messages,
                "assistant_response": response,
                "official_response": official_response,
                "chain_of_thought": reasoning,
                "final_answer": action,
                "label": int(verdict),
                "label_source": normalization["label_source"],
                "annotation_protocol": normalization["annotation_protocol"],
                "annotation_metadata": {
                    "verification_result": verdict,
                    "verifier": run_manifest["verifier"],
                    "artifact_sha256": artifact.sha256,
                    "artifact_line_number": record["artifact_line_number"],
                    "response_index": response_index,
                    "response_sha256": response_sha256,
                },
                "eligible_for_main_study": eligible_for_main_study,
                "construct_name": normalization["construct_name"],
            }
            validate_normalized_monitorbench_row(
                row, adapter=adapter, adapter_sha256=adapter_sha256
            )
            rows.append(row)
    return rows


def _parse_artifact_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--artifact must use TASK_ID=PATH")
    task_id, raw_path = value.split("=", 1)
    if not task_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--artifact must use TASK_ID=PATH")
    return task_id.strip(), Path(raw_path.strip())


def _import_manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_sha256"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize pinned official MonitorBench .tested.jsonl artifacts into the "
            "test-only verified-outcome transfer family. This does not run MonitorBench."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--artifact",
        action="append",
        type=_parse_artifact_argument,
        metavar="TASK_ID=PATH",
        help="Repeat for each official evaluated_llm_<stress>_n=<N>.tested.jsonl file.",
    )
    inputs.add_argument(
        "--results_root",
        help=(
            "Discover official artifacts below one evaluated model's MonitorBench results "
            "root using <task>/inference_results/*.tested.jsonl."
        ),
    )
    parser.add_argument(
        "--adapter", default=str(DEFAULT_MONITORBENCH_ADAPTER)
    )
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--run_manifest", required=True)
    parser.add_argument(
        "--falsification_registry",
        default="experiments/protocol/falsification_registry.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow_incomplete_suite",
        action="store_true",
        help=(
            "Allow an incomplete official task/stress matrix. Rows are marked ineligible "
            "for the main study."
        ),
    )
    args = parser.parse_args()

    adapter, adapter_sha256 = load_monitorbench_adapter(Path(args.adapter))
    _, source_manifest_sha256 = validate_monitorbench_source_manifest(
        Path(args.source_manifest), adapter=adapter
    )
    run_manifest, run_manifest_sha256 = load_monitorbench_run_manifest(
        Path(args.run_manifest),
        adapter=adapter,
        adapter_sha256=adapter_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )
    registry, registry_sha256 = load_falsification_registry(
        Path(args.falsification_registry)
    )

    artifact_arguments = list(args.artifact or [])
    if args.results_root:
        results_root = Path(args.results_root)
        if not results_root.is_dir():
            raise FileNotFoundError(
                f"Missing MonitorBench results root: {results_root}"
            )
        for artifact_path in sorted(
            results_root.rglob("evaluated_llm_*_n=*.tested.jsonl")
        ):
            if artifact_path.parent.name != "inference_results":
                continue
            artifact_arguments.append(
                (artifact_path.parent.parent.name, artifact_path)
            )
        if not artifact_arguments:
            raise ValueError(
                f"No official MonitorBench tested artifacts found below {results_root}"
            )

    artifacts: list[MonitorBenchArtifact] = []
    observed: set[tuple[str, str]] = set()
    for task_id, artifact_path in artifact_arguments:
        artifact = load_monitorbench_tested_artifact(
            artifact_path,
            task_id=task_id,
            adapter=adapter,
            expected_rollout_number=run_manifest["generation"]["rollout_number"],
        )
        key = (artifact.task_id, artifact.stress_test)
        if key in observed:
            raise ValueError(f"Duplicate MonitorBench artifact for {key}")
        observed.add(key)
        artifacts.append(artifact)

    expected = expected_monitorbench_artifacts(adapter)
    missing = sorted(expected.difference(observed))
    extra = sorted(observed.difference(expected))
    complete_suite = not missing and not extra
    if not complete_suite and not args.allow_incomplete_suite:
        raise ValueError(
            "Main-study MonitorBench import requires the complete official task/stress "
            f"matrix; missing={missing[:10]}, extra={extra[:10]}"
        )
    eligible = complete_suite
    rows = [
        row
        for artifact in sorted(
            artifacts, key=lambda item: (item.task_id, item.stress_test)
        )
        for row in normalize_monitorbench_artifact(
            artifact,
            adapter=adapter,
            adapter_sha256=adapter_sha256,
            source_manifest_sha256=source_manifest_sha256,
            run_manifest=run_manifest,
            run_manifest_sha256=run_manifest_sha256,
            falsification_registry=registry,
            falsification_registry_sha256=registry_sha256,
            eligible_for_main_study=eligible,
        )
    ]
    if not rows:
        raise ValueError("MonitorBench import produced no rollouts")
    label_counts = Counter(int(row["label"]) for row in rows)
    scenario_labels: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        scenario_labels[row["scenario_id"]].add(int(row["label"]))
    matched_scenarios = sum(labels == {0, 1} for labels in scenario_labels.values())
    if complete_suite and (set(label_counts) != {0, 1} or matched_scenarios < 1):
        raise ValueError(
            "Complete MonitorBench import lacks both verifier outcomes within any exact prompt"
        )

    output_path = Path(args.output)
    _atomic_write_jsonl(output_path, rows)
    import_manifest: dict[str, Any] = {
        "schema_version": IMPORT_MANIFEST_SCHEMA_VERSION,
        "adapter_id": adapter["adapter_id"],
        "adapter_sha256": adapter_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "run_id": run_manifest["run_id"],
        "run_manifest_sha256": run_manifest_sha256,
        "falsification_registry_sha256": registry_sha256,
        "complete_official_suite": complete_suite,
        "eligible_for_main_study": eligible,
        "missing_task_stress_pairs": [list(item) for item in missing],
        "n_artifacts": len(artifacts),
        "n_rows": len(rows),
        "n_scenarios": len(scenario_labels),
        "n_exact_prompt_matched_scenarios": matched_scenarios,
        "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "artifacts": [
            {
                "task_id": artifact.task_id,
                "stress_test": artifact.stress_test,
                "rollout_number": artifact.rollout_number,
                "sha256": artifact.sha256,
                "n_records": len(artifact.records),
                "path": str(artifact.path),
            }
            for artifact in sorted(
                artifacts, key=lambda item: (item.task_id, item.stress_test)
            )
        ],
        "output_file": str(output_path),
        "output_sha256": file_sha256(output_path),
    }
    import_manifest["manifest_sha256"] = content_hash(
        _import_manifest_payload(import_manifest)
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    _atomic_write_json(manifest_path, import_manifest)
    print(
        f"saved {output_path} ({len(rows)} rows; {matched_scenarios} exact-prompt "
        f"matched scenarios)"
    )
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
