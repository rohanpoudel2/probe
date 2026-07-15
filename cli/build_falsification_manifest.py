from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data.falsification import (
    FALSIFICATION_EVALUATION_SCHEMA_VERSION,
    SHIFT_AXES,
    falsification_signature,
    file_sha256,
    has_heldout_shift,
    load_falsification_registry,
    prompt_messages_sha256,
    validate_falsification_metadata,
)
from data.rollout_schema import content_hash


COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")


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
    if not rows:
        raise ValueError(f"No labeled rows found in {path}")
    return rows


def _identity(rows: list[dict[str, Any]]) -> dict[str, str]:
    identities = {
        (
            str(row.get("model_id", "")),
            str(row.get("model_revision", "")),
            str(row.get("tokenizer_revision", "")),
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError(
            "Falsification data must use exactly one monitored model identity"
        )
    model_id, model_revision, tokenizer_revision = identities.pop()
    if (
        not model_id
        or not COMMIT_RE.fullmatch(model_revision)
        or not COMMIT_RE.fullmatch(tokenizer_revision)
    ):
        raise ValueError(
            "Falsification data lacks immutable model/tokenizer provenance"
        )
    return {
        "monitored_model_id": model_id,
        "monitored_model_revision": model_revision,
        "monitored_tokenizer_revision": tokenizer_revision,
    }


def _example_record(
    row: dict[str, Any],
    *,
    task_name: str,
    registry: dict[str, Any],
    registry_sha256: str,
) -> dict[str, Any]:
    if (
        row.get("data_origin") != "on_policy_generation"
        or row.get("generated_by_model") is not True
    ):
        raise ValueError(
            f"Falsification example {row.get('example_id')} is not on-policy"
        )
    label = row.get("label")
    if label not in {0, 1}:
        raise ValueError(
            f"Falsification example {row.get('example_id')} lacks a binary label"
        )
    example_id = str(row.get("example_id") or row.get("rollout_id") or "").strip()
    scenario_id = str(row.get("scenario_id", "")).strip()
    group_id = str(row.get("question_id") or row.get("group_id") or "").strip()
    condition = str(row.get("condition", "")).strip()
    split = str(row.get("protocol_split", "")).strip()
    if not all((example_id, scenario_id, group_id, condition)) or split not in {
        "train",
        "calibration",
        "eval",
        "test",
    }:
        raise ValueError(
            f"Falsification example {example_id!r} has incomplete identifiers"
        )
    scenario_metadata = row.get("metadata")
    if not isinstance(scenario_metadata, dict):
        raise ValueError(f"Falsification example {example_id} lacks scenario metadata")
    falsification = validate_falsification_metadata(
        scenario_metadata.get("falsification"),
        registry=registry,
        registry_sha256=registry_sha256,
        task_name=task_name,
    )
    if has_heldout_shift(falsification) and split != "test":
        raise ValueError(
            f"Held-out shift example {example_id} is assigned to {split}, not test"
        )
    return {
        "example_id": example_id,
        "scenario_id": scenario_id,
        "group_id": group_id,
        "condition": condition,
        "protocol_split": split,
        "label": int(label),
        "prompt_sha256": prompt_messages_sha256(row),
        "row_sha256": content_hash(row),
        "shift_signature": falsification_signature(falsification),
        "axes": falsification["axes"],
    }


def _hard_negative_pairs(
    examples: list[dict[str, Any]],
    *,
    task_name: str,
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    task_protocol = registry["tasks"][task_name]["hard_negative"]
    if task_protocol["enabled"] is not True:
        return []
    triggers = set(task_protocol["trigger_conditions"])
    candidates = [
        example
        for example in examples
        if example["protocol_split"] == "test" and example["condition"] in triggers
    ]
    strata: dict[tuple[str, str, str], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for example in candidates:
        key = (
            example["scenario_id"],
            example["shift_signature"],
            example["prompt_sha256"],
        )
        strata[key][example["label"]].append(example)
    pairs: list[dict[str, Any]] = []
    for (scenario_id, signature, prompt_hash), labels in sorted(strata.items()):
        positives = sorted(labels[1], key=lambda row: row["example_id"])
        negatives = sorted(labels[0], key=lambda row: row["example_id"])
        for positive, negative in zip(positives, negatives):
            if positive["group_id"] != negative["group_id"]:
                raise ValueError(
                    "Exact-scenario hard-negative candidates disagree on group"
                )
            pair_payload = {
                "task_name": task_name,
                "scenario_id": scenario_id,
                "group_id": positive["group_id"],
                "shift_signature": signature,
                "prompt_sha256": prompt_hash,
                "positive_example_id": positive["example_id"],
                "negative_example_id": negative["example_id"],
                "positive_row_sha256": positive["row_sha256"],
                "negative_row_sha256": negative["row_sha256"],
            }
            pairs.append(
                {
                    "pair_id": content_hash(pair_payload),
                    "match_type": "exact_trigger_prompt",
                    **pair_payload,
                }
            )
    return pairs


def build_falsification_manifest(
    rows: list[dict[str, Any]],
    *,
    source_path: Path,
    registry: dict[str, Any],
    registry_sha256: str,
    minimum_hard_negative_pairs: int,
    model_name: str,
) -> dict[str, Any]:
    if not model_name.strip():
        raise ValueError("Falsification manifest requires the benchmark model name")
    task_names = {str(row.get("task_family", "")) for row in rows}
    if len(task_names) != 1:
        raise ValueError(
            "A falsification manifest must describe exactly one task family"
        )
    task_name = task_names.pop()
    if task_name not in registry["tasks"]:
        raise ValueError(
            f"Task {task_name!r} is absent from the falsification registry"
        )
    examples = [
        _example_record(
            row,
            task_name=task_name,
            registry=registry,
            registry_sha256=registry_sha256,
        )
        for row in rows
    ]
    example_ids = [example["example_id"] for example in examples]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("Falsification data contains duplicate example IDs")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        group_splits[example["group_id"]].add(example["protocol_split"])
    leaking = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking:
        raise ValueError(
            f"Falsification data leaks groups across splits: {leaking[:5]}"
        )
    pairs = _hard_negative_pairs(examples, task_name=task_name, registry=registry)
    hard_negative_enabled = registry["tasks"][task_name]["hard_negative"]["enabled"]
    if hard_negative_enabled and len(pairs) < minimum_hard_negative_pairs:
        raise ValueError(
            f"Task {task_name} has {len(pairs)} exact-prompt hard-negative pairs; "
            f"requires {minimum_hard_negative_pairs}"
        )
    axis_counts: Counter[tuple[str, str, str, str]] = Counter()
    for example in examples:
        for axis in SHIFT_AXES:
            entry = example["axes"][axis]
            axis_counts[
                (
                    axis,
                    str(entry["value"]),
                    str(entry["role"]),
                    example["protocol_split"],
                )
            ] += 1
    summary = {
        "n_examples": len(examples),
        "n_groups": len(group_splits),
        "n_hard_negative_pairs": len(pairs),
        "n_hard_negative_groups": len({pair["group_id"] for pair in pairs}),
        "axis_counts": [
            {
                "axis": key[0],
                "value": key[1],
                "role": key[2],
                "protocol_split": key[3],
                "count": count,
            }
            for key, count in sorted(axis_counts.items())
        ],
    }
    manifest: dict[str, Any] = {
        "schema_version": FALSIFICATION_EVALUATION_SCHEMA_VERSION,
        "registry_id": registry["registry_id"],
        "registry_sha256": registry_sha256,
        "task_name": task_name,
        "model": model_name,
        **_identity(rows),
        "source_data_file": str(source_path.resolve()),
        "source_data_sha256": file_sha256(source_path),
        "hard_negative_match_type": "exact_trigger_prompt",
        "examples": examples,
        "hard_negative_pairs": pairs,
        "summary": summary,
    }
    manifest["manifest_id"] = content_hash(
        {
            "registry_sha256": registry_sha256,
            "task_name": task_name,
            "source_data_sha256": manifest["source_data_sha256"],
            "monitored_model_revision": manifest["monitored_model_revision"],
        }
    )
    manifest["manifest_sha256"] = content_hash(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build immutable shift slices and exact-prompt hard-negative pairs"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--minimum_hard_negative_pairs", type=int, default=None)
    args = parser.parse_args()

    registry, registry_sha256 = load_falsification_registry(Path(args.registry))
    minimum = (
        int(args.minimum_hard_negative_pairs)
        if args.minimum_hard_negative_pairs is not None
        else int(registry["hard_negative_protocol"]["min_pairs_pilot"])
    )
    if minimum < 0:
        raise ValueError("minimum_hard_negative_pairs cannot be negative")
    data_path = Path(args.data)
    manifest = build_falsification_manifest(
        _read_jsonl(data_path),
        source_path=data_path,
        registry=registry,
        registry_sha256=registry_sha256,
        minimum_hard_negative_pairs=minimum,
        model_name=args.model_name,
    )
    _atomic_write_json(Path(args.output), manifest)
    print(
        f"saved {manifest['summary']['n_examples']} indexed examples and "
        f"{manifest['summary']['n_hard_negative_pairs']} hard-negative pairs to {args.output}"
    )


if __name__ == "__main__":
    main()
