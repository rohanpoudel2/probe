from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data.reference_traffic import (
    REFERENCE_LABEL_SOURCE,
    REFERENCE_PROTOCOL,
    validate_reference_annotation_metadata,
)
from data.falsification import (
    SHIFT_AXES,
    has_heldout_shift,
    load_falsification_registry,
    validate_falsification_metadata,
)
from data.generation_confidence import confidence_feature_vector
from data.rollout_schema import content_hash, validate_messages


PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            rows.append(row)
    return rows


def _normalized_prompt(messages: list[dict[str, str]]) -> str:
    text = "\n".join(
        message["content"] for message in messages if message.get("role") != "assistant"
    )
    return " ".join(text.lower().split())


def audit_rows(
    rows: list[dict[str, Any]],
    *,
    reference_traffic: bool = False,
    require_confidence_trace: bool = False,
    falsification_bundle: tuple[dict[str, Any], str] | None = None,
    require_falsification: bool = False,
) -> dict[str, Any]:
    if require_falsification and falsification_bundle is None:
        raise ValueError("require_falsification needs a falsification registry")
    errors: list[str] = []
    warnings: list[str] = []
    rollout_ids: set[str] = set()
    scenario_ids: set[str] = set()
    group_labels: dict[str, set[int]] = defaultdict(set)
    group_conditions: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    group_counts: Counter[str] = Counter()
    prompt_groups: dict[str, set[str]] = defaultdict(set)
    label_sources: Counter[str] = Counter()
    models: Counter[str] = Counter()
    confidence_traces = 0
    shift_counts: Counter[tuple[str, str, str, str]] = Counter()

    for index, row in enumerate(rows):
        prefix = f"row {index}"
        rollout_id = str(row.get("rollout_id", "")).strip()
        scenario_id = str(row.get("scenario_id", "")).strip()
        group_id = str(row.get("group_id") or row.get("question_id") or "").strip()
        if not rollout_id or not scenario_id or not group_id:
            errors.append(
                f"{prefix}: rollout_id, scenario_id, and group_id/question_id are required"
            )
            continue
        if rollout_id in rollout_ids:
            errors.append(f"{prefix}: duplicate rollout_id {rollout_id}")
        rollout_ids.add(rollout_id)
        scenario_ids.add(scenario_id)
        group_counts[group_id] += 1

        if (
            row.get("data_origin") != "on_policy_generation"
            or row.get("generated_by_model") is not True
        ):
            errors.append(f"{prefix}: not marked as on-policy model generation")
        label = row.get("label")
        if label not in {0, 1}:
            errors.append(f"{prefix}: label must be 0 or 1")
        else:
            group_labels[group_id].add(int(label))
        condition = str(row.get("condition", "")).strip()
        if not condition:
            errors.append(f"{prefix}: condition is required")
        else:
            group_conditions[group_id].add(condition)
        protocol_split = str(row.get("protocol_split", "")).strip()
        if protocol_split not in {"train", "calibration", "eval", "test"}:
            errors.append(f"{prefix}: invalid or missing protocol_split")
        else:
            group_splits[group_id].add(protocol_split)

        if falsification_bundle is not None and not reference_traffic:
            registry, registry_sha256 = falsification_bundle
            task_name = str(row.get("task_family", ""))
            scenario_metadata = row.get("metadata")
            try:
                if not isinstance(scenario_metadata, dict):
                    raise ValueError("scenario metadata must be an object")
                falsification = validate_falsification_metadata(
                    scenario_metadata.get("falsification"),
                    registry=registry,
                    registry_sha256=registry_sha256,
                    task_name=task_name,
                )
                if has_heldout_shift(falsification) and protocol_split != "test":
                    raise ValueError(
                        "held-out shift is assigned outside the test split"
                    )
                for axis in SHIFT_AXES:
                    entry = falsification["axes"][axis]
                    shift_counts[
                        (
                            axis,
                            str(entry["value"]),
                            str(entry["role"]),
                            protocol_split,
                        )
                    ] += 1
            except ValueError as err:
                errors.append(f"{prefix}: invalid falsification metadata: {err}")
        elif require_falsification and not reference_traffic:
            errors.append(f"{prefix}: missing required falsification metadata")

        label_source = str(row.get("label_source", "")).strip()
        annotation_protocol = str(row.get("annotation_protocol", "")).strip()
        if not label_source or not annotation_protocol:
            errors.append(
                f"{prefix}: label_source and annotation_protocol are required"
            )
        label_sources[label_source] += 1
        if reference_traffic:
            if protocol_split not in {"calibration", "test"}:
                errors.append(f"{prefix}: invalid reference traffic partition")
            if label != 0:
                errors.append(
                    f"{prefix}: reference rows must use membership value 0"
                )
            if (
                label_source != REFERENCE_LABEL_SOURCE
                or annotation_protocol != REFERENCE_PROTOCOL
            ):
                errors.append(f"{prefix}: invalid reference traffic contract")
            try:
                validate_reference_annotation_metadata(
                    row, row.get("annotation_metadata")
                )
            except ValueError as err:
                errors.append(f"{prefix}: invalid reference metadata: {err}")

        revision = str(row.get("model_revision", "")).strip()
        if not PINNED_REVISION_RE.fullmatch(revision):
            errors.append(
                f"{prefix}: model_revision is not an immutable commit-like identifier"
            )
        models[f"{row.get('model_id')}@{revision}"] += 1
        generation = row.get("generation")
        has_confidence = (
            isinstance(generation, dict) and "confidence_trace" in generation
        )
        if has_confidence or require_confidence_trace:
            try:
                confidence_feature_vector(generation)
                confidence_traces += 1
            except ValueError as err:
                errors.append(
                    f"{prefix}: invalid or missing generation confidence: {err}"
                )
        tokenizer_revision = str(row.get("tokenizer_revision", "")).strip()
        if not PINNED_REVISION_RE.fullmatch(tokenizer_revision):
            errors.append(
                f"{prefix}: tokenizer_revision is not an immutable commit-like identifier"
            )
        provenance = row.get("provenance") or {}
        if not isinstance(provenance, dict):
            errors.append(f"{prefix}: provenance must be an object")
        else:
            if provenance.get("code_dirty") is not False:
                errors.append(
                    f"{prefix}: rollout was generated from dirty or unknown code state"
                )
            if not PINNED_REVISION_RE.fullmatch(str(provenance.get("code_commit", ""))):
                errors.append(f"{prefix}: provenance lacks an immutable code_commit")
            for hash_key in ("chat_template_sha256", "scenario_file_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(hash_key, ""))):
                    errors.append(f"{prefix}: provenance lacks a valid {hash_key}")

        try:
            prompt_messages = validate_messages(
                row.get("prompt_messages"), allow_assistant=False
            )
            full_messages = validate_messages(row.get("messages"), allow_assistant=True)
            if full_messages[:-1] != prompt_messages:
                errors.append(
                    f"{prefix}: full messages do not extend prompt_messages exactly"
                )
            if full_messages[-1]["content"] != row.get("response_text"):
                errors.append(f"{prefix}: assistant message differs from response_text")
            prompt_groups[content_hash(_normalized_prompt(prompt_messages))].add(
                group_id
            )
        except ValueError as err:
            errors.append(f"{prefix}: invalid messages: {err}")

    cross_group_prompt_duplicates = {
        prompt_hash: sorted(groups)
        for prompt_hash, groups in prompt_groups.items()
        if len(groups) > 1
    }
    if cross_group_prompt_duplicates:
        errors.append(
            f"Exact normalized prompts occur across {len(cross_group_prompt_duplicates)} different group IDs"
        )

    matched_groups = sorted(
        group for group, labels in group_labels.items() if labels == {0, 1}
    )
    unmatched_groups = sorted(
        group for group, labels in group_labels.items() if labels != {0, 1}
    )
    if reference_traffic:
        non_negative_groups = [
            group for group, labels in group_labels.items() if labels != {0}
        ]
        if non_negative_groups:
            errors.append(
                "Reference traffic requires membership value 0 for every row"
            )
        repeated_groups = [group for group, count in group_counts.items() if count != 1]
        if repeated_groups:
            errors.append(
                "Reference traffic requires exactly one rollout per independent group"
            )
    else:
        if not matched_groups:
            errors.append(
                "No group contains both observed labels; matched few-shot training is impossible"
            )
        elif unmatched_groups:
            warnings.append(
                f"{len(unmatched_groups)} groups do not contain both labels and cannot enter matched few-shot training"
            )
    single_condition_groups = [
        group for group, values in group_conditions.items() if len(values) < 2
    ]
    if single_condition_groups:
        warnings.append(
            f"{len(single_condition_groups)} groups contain only one experimental condition"
        )
    leaking_split_groups = [
        group for group, values in group_splits.items() if len(values) != 1
    ]
    if leaking_split_groups:
        errors.append(
            f"{len(leaking_split_groups)} groups span multiple protocol splits"
        )

    return {
        "status": "fail" if errors else "pass",
        "n_rows": len(rows),
        "n_rollouts": len(rollout_ids),
        "n_scenarios": len(scenario_ids),
        "n_groups": len(group_labels),
        "n_matched_groups": len(matched_groups),
        "n_unmatched_groups": len(unmatched_groups),
        "label_sources": dict(label_sources),
        "models": dict(models),
        "n_valid_generation_confidence_traces": confidence_traces,
        "shift_counts": [
            {
                "axis": key[0],
                "value": key[1],
                "role": key[2],
                "protocol_split": key[3],
                "count": count,
            }
            for key, count in sorted(shift_counts.items())
        ],
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit labeled rollout data before activation extraction"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--reference_traffic", action="store_true")
    parser.add_argument("--require_confidence_trace", action="store_true")
    parser.add_argument("--falsification_registry", default=None)
    parser.add_argument("--require_falsification", action="store_true")
    args = parser.parse_args()

    falsification_bundle = (
        load_falsification_registry(Path(args.falsification_registry))
        if args.falsification_registry
        else None
    )
    report = audit_rows(
        _read_jsonl(Path(args.data)),
        reference_traffic=args.reference_traffic,
        require_confidence_trace=args.require_confidence_trace,
        falsification_bundle=falsification_bundle,
        require_falsification=args.require_falsification,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved {output}")
    print(rendered)
    if report["status"] != "pass":
        raise RuntimeError("Rollout dataset audit failed")


if __name__ == "__main__":
    main()
