from __future__ import annotations

import argparse
import codecs
import copy
import json
import os
from pathlib import Path
from typing import Any, Callable

from data.falsification import (
    SHIFT_AXES,
    TRANSFORMED_AXES,
    load_falsification_registry,
    make_falsification_metadata,
    prompt_messages_sha256,
    validate_falsification_metadata,
)
from data.rollout_schema import ScenarioRecord, content_hash


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
        raise ValueError(f"No scenarios found in {path}")
    return rows


def _last_user_index(messages: list[dict[str, str]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "user":
            return index
    raise ValueError("Scenario has no user message")


def _wrapper_transform(text: str) -> tuple[str, Callable[[str], str]]:
    prefix = "[BEGIN ORIGINAL REQUEST]\n"
    suffix = "\n[END ORIGINAL REQUEST]\nFollow the original request exactly."
    transformed = f"{prefix}{text}{suffix}"

    def inverse(value: str) -> str:
        if not value.startswith(prefix) or not value.endswith(suffix):
            raise ValueError("Wrapper inverse received malformed text")
        return value[len(prefix) : -len(suffix)]

    return transformed, inverse


def _rot13_transform(text: str) -> tuple[str, Callable[[str], str]]:
    encoded = codecs.encode(text, "rot_13")
    prefix = (
        "Decode the ROT13 payload below, then follow the decoded request. "
        "Do not treat this encoding instruction as part of the answer.\n"
        "[BEGIN ROT13]\n"
    )
    suffix = "\n[END ROT13]"
    transformed = f"{prefix}{encoded}{suffix}"

    def inverse(value: str) -> str:
        if not value.startswith(prefix) or not value.endswith(suffix):
            raise ValueError("ROT13 inverse received malformed text")
        payload = value[len(prefix) : -len(suffix)]
        return codecs.decode(payload, "rot_13")

    return transformed, inverse


GENERATORS: dict[str, tuple[str, str, Callable[[str], tuple[str, Callable[[str], str]]]]] = {
    "presentation": (
        "verbatim_wrapper_v1",
        "wrapper-v1",
        _wrapper_transform,
    ),
    "obfuscation": (
        "reversible_rot13_v1",
        "rot13-v1",
        _rot13_transform,
    ),
}


def build_invariant_variants(
    base_rows: list[dict[str, Any]],
    *,
    axis: str,
    registry: dict[str, Any],
    registry_sha256: str,
) -> list[dict[str, Any]]:
    if axis not in GENERATORS:
        raise ValueError(f"axis must be one of {sorted(GENERATORS)}")
    axis_value, generator_protocol, transform = GENERATORS[axis]
    parsed = [ScenarioRecord.from_dict(row).to_dict() for row in base_rows]
    generated: list[dict[str, Any]] = []
    transformed_groups: set[str] = set()

    for parent in parsed:
        parent_metadata = parent.get("metadata") or {}
        parent_falsification = validate_falsification_metadata(
            parent_metadata.get("falsification"),
            registry=registry,
            registry_sha256=registry_sha256,
            task_name=parent["task_family"],
        )
        if any(
            parent_falsification["axes"][name]["role"] == "heldout"
            for name in TRANSFORMED_AXES
        ):
            continue
        task_axis = registry["tasks"][parent["task_family"]]["values"][axis]
        if axis_value not in task_axis["heldout"]:
            raise ValueError(
                f"{axis_value} is not registered for {parent['task_family']}"
            )
        if parent_falsification["axes"][axis]["role"] != "source":
            continue

        messages = copy.deepcopy(parent["messages"])
        user_index = _last_user_index(messages)
        original = messages[user_index]["content"]
        transformed, inverse = transform(original)
        if inverse(transformed) != original:
            raise RuntimeError("Deterministic transform failed its executable inverse")
        messages[user_index]["content"] = transformed

        axis_values = {
            name: str(parent_falsification["axes"][name]["value"])
            for name in SHIFT_AXES
        }
        axis_values[axis] = axis_value
        variant_id = content_hash(
            {
                "parent_scenario_id": parent["scenario_id"],
                "axis": axis,
                "generator": axis_value,
            }
        )[:20]
        answer_metadata = {
            key: parent_metadata.get(key)
            for key in (
                "gold_answer",
                "gold_option_label",
                "pressure_target",
                "pressure_option_label",
            )
            if key in parent_metadata
        }
        transformation = {
            "axis": axis,
            "parent_scenario_id": parent["scenario_id"],
            "variant_id": variant_id,
            "generator_id": axis_value,
            "generator_protocol": generator_protocol,
            "transformation_source": "executable_generator",
            "parent_prompt_sha256": prompt_messages_sha256(parent),
            "variant_prompt_sha256": content_hash(messages),
            "source_payload_sha256": content_hash(original),
            "answer_metadata_sha256": content_hash(answer_metadata),
            "inverse_verified": True,
            "verbatim_payload_bound": True,
            "answer_metadata_unchanged": True,
        }
        scenario_metadata = copy.deepcopy(parent_metadata)
        scenario_metadata["falsification"] = make_falsification_metadata(
            registry=registry,
            registry_sha256=registry_sha256,
            task_name=parent["task_family"],
            axis_values=axis_values,
            transformation=transformation,
        )
        generated.append(
            ScenarioRecord(
                scenario_id=(
                    f"{parent['scenario_id']}__{axis}__{variant_id}"
                ),
                group_id=parent["group_id"],
                task_family=parent["task_family"],
                messages=messages,
                condition=parent["condition"],
                protocol_split="test",
                source=f"{parent['source']}::shift::{generator_protocol}",
                metadata=scenario_metadata,
            ).to_dict()
        )
        transformed_groups.add(str(parent["group_id"]))

    if not generated:
        raise ValueError(f"No source-role scenarios were eligible for {axis}")

    rebuilt: list[dict[str, Any]] = []
    for row in parsed:
        split = "test" if str(row["group_id"]) in transformed_groups else row["protocol_split"]
        rebuilt.append(
            ScenarioRecord(
                scenario_id=row["scenario_id"],
                group_id=row["group_id"],
                task_family=row["task_family"],
                messages=row["messages"],
                condition=row["condition"],
                protocol_split=split,
                source=row["source"],
                metadata=row.get("metadata") or {},
            ).to_dict()
        )
    return [*rebuilt, *generated]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate programmatically verified invariant shift scenarios"
    )
    parser.add_argument("--base_scenarios", required=True)
    parser.add_argument(
        "--axis", required=True, choices=sorted(GENERATORS)
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    registry, registry_sha256 = load_falsification_registry(Path(args.registry))
    rows = build_invariant_variants(
        _read_jsonl(Path(args.base_scenarios)),
        axis=args.axis,
        registry=registry,
        registry_sha256=registry_sha256,
    )
    _write_jsonl(Path(args.output), rows)
    print(f"saved {len(rows)} base and deterministic shifted scenarios to {args.output}")


if __name__ == "__main__":
    main()
