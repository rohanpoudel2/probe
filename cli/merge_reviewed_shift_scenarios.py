from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from data.falsification import (
    SHIFT_AXES,
    TRANSFORMED_AXES,
    file_sha256,
    load_falsification_registry,
    make_falsification_metadata,
    prompt_messages_sha256,
    validate_falsification_metadata,
)
from data.rollout_schema import ScenarioRecord, content_hash, validate_messages


ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
PROHIBITED_OUTCOME_FIELDS = {
    "label",
    "assistant_response",
    "response_text",
    "final_answer",
    "reasoning",
    "chain_of_thought",
}


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
        raise ValueError(f"No records found in {path}")
    return rows


def _review_summary(
    ratings: Any,
    *,
    variant_prompt_sha256: str,
    registry: dict[str, Any],
    transformation_author_id: str,
) -> dict[str, Any]:
    if not isinstance(ratings, list):
        raise ValueError("Shift variant ratings must be a list")
    required = registry["transformation_review"]["required_boolean_decisions"]
    min_raters = int(registry["transformation_review"]["min_independent_raters"])
    reviewer_ids: list[str] = []
    rating_ids: list[str] = []
    for rating in ratings:
        if not isinstance(rating, dict):
            raise ValueError("Every shift-variant rating must be an object")
        reviewer_id = str(rating.get("reviewer_id", "")).strip()
        rating_id = str(rating.get("rating_id", "")).strip()
        if not reviewer_id or not rating_id:
            raise ValueError(
                "Every shift-variant rating requires reviewer_id and rating_id"
            )
        if reviewer_id == transformation_author_id:
            raise ValueError(
                "The transformation author cannot review their own variant"
            )
        if rating.get("variant_prompt_sha256") != variant_prompt_sha256:
            raise ValueError(
                f"Shift-variant rating {rating_id} has a stale prompt hash"
            )
        for decision in required:
            if rating.get(decision) is not True:
                raise ValueError(f"Shift-variant rating {rating_id} failed {decision}")
        reviewer_ids.append(reviewer_id)
        rating_ids.append(rating_id)
    if len(set(reviewer_ids)) < min_raters or len(set(reviewer_ids)) != len(
        reviewer_ids
    ):
        raise ValueError("Shift variant lacks enough distinct independent reviewers")
    if len(set(rating_ids)) != len(rating_ids):
        raise ValueError("Shift variant contains duplicate rating IDs")
    return {
        "protocol": registry["transformation_review"]["protocol"],
        "independent_rater_ids": sorted(reviewer_ids),
        "rating_ids": sorted(rating_ids),
        **{decision: True for decision in required},
    }


def _rebuild_with_split(row: dict[str, Any], split: str) -> dict[str, Any]:
    return ScenarioRecord(
        scenario_id=row["scenario_id"],
        group_id=row["group_id"],
        task_family=row["task_family"],
        messages=row["messages"],
        condition=row["condition"],
        protocol_split=split,
        source=row["source"],
        metadata=row.get("metadata") or {},
    ).to_dict()


def merge_reviewed_variants(
    base_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    *,
    registry: dict[str, Any],
    registry_sha256: str,
    review_file_sha256: str,
) -> list[dict[str, Any]]:
    parsed_base = [ScenarioRecord.from_dict(row).to_dict() for row in base_rows]
    base_by_id = {row["scenario_id"]: row for row in parsed_base}
    if len(base_by_id) != len(parsed_base):
        raise ValueError("Base scenario inventory contains duplicate scenario IDs")
    variant_ids: set[str] = set()
    generated_ids: set[str] = set(base_by_id)
    transformed_groups: set[str] = set()
    generated: list[dict[str, Any]] = []

    for variant in variant_rows:
        prohibited = sorted(
            key for key in PROHIBITED_OUTCOME_FIELDS if variant.get(key) is not None
        )
        if prohibited:
            raise ValueError(
                f"Shift variant contains prohibited outcome fields {prohibited}"
            )
        variant_id = str(variant.get("variant_id", "")).strip()
        parent_id = str(variant.get("parent_scenario_id", "")).strip()
        axis = str(variant.get("axis", "")).strip()
        axis_value = str(variant.get("axis_value", "")).strip()
        if ID_RE.fullmatch(variant_id) is None or variant_id in variant_ids:
            raise ValueError(f"Invalid or duplicate shift variant_id {variant_id!r}")
        if parent_id not in base_by_id:
            raise ValueError(f"Unknown parent_scenario_id {parent_id!r}")
        if axis not in TRANSFORMED_AXES or not axis_value:
            raise ValueError("Reviewed variants must target paraphrase or obfuscation")
        parent = base_by_id[parent_id]
        parent_metadata = parent.get("metadata") or {}
        parent_falsification = validate_falsification_metadata(
            parent_metadata.get("falsification"),
            registry=registry,
            registry_sha256=registry_sha256,
            task_name=parent["task_family"],
        )
        if parent_falsification["axes"][axis]["role"] != "source":
            raise ValueError("A shifted variant must descend from a source-role parent")
        messages = validate_messages(variant.get("messages"), allow_assistant=False)
        variant_prompt_sha256 = content_hash(messages)
        parent_prompt_sha256 = prompt_messages_sha256(parent)
        if variant_prompt_sha256 == parent_prompt_sha256:
            raise ValueError("Shifted prompt is identical to its parent")
        transformation_protocol = str(
            variant.get("transformation_protocol", "")
        ).strip()
        transformation_source = str(variant.get("transformation_source", "")).strip()
        transformation_author_id = str(
            variant.get("transformation_author_id", "")
        ).strip()
        if (
            not transformation_protocol
            or not transformation_source
            or not transformation_author_id
        ):
            raise ValueError(
                "Shift variant requires transformation protocol, source, and author ID"
            )
        review = _review_summary(
            variant.get("ratings"),
            variant_prompt_sha256=variant_prompt_sha256,
            registry=registry,
            transformation_author_id=transformation_author_id,
        )
        axis_values = {
            name: str(parent_falsification["axes"][name]["value"])
            for name in SHIFT_AXES
        }
        axis_values[axis] = axis_value
        transformation = {
            "axis": axis,
            "parent_scenario_id": parent_id,
            "variant_id": variant_id,
            "transformation_protocol": transformation_protocol,
            "transformation_source": transformation_source,
            "transformation_author_id": transformation_author_id,
            "parent_prompt_sha256": parent_prompt_sha256,
            "variant_prompt_sha256": variant_prompt_sha256,
            "review_file_sha256": review_file_sha256,
            "review": review,
        }
        scenario_metadata = copy.deepcopy(parent_metadata)
        scenario_metadata["falsification"] = make_falsification_metadata(
            registry=registry,
            registry_sha256=registry_sha256,
            task_name=parent["task_family"],
            axis_values=axis_values,
            transformation=transformation,
        )
        scenario_id = f"{parent_id}__{axis}__{variant_id}__{variant_prompt_sha256[:12]}"
        if scenario_id in generated_ids:
            raise ValueError(f"Duplicate generated shift scenario {scenario_id}")
        record = ScenarioRecord(
            scenario_id=scenario_id,
            group_id=parent["group_id"],
            task_family=parent["task_family"],
            messages=messages,
            condition=parent["condition"],
            protocol_split="test",
            source=f"{parent['source']}::shift::{transformation_protocol}",
            metadata=scenario_metadata,
        )
        generated.append(record.to_dict())
        generated_ids.add(scenario_id)
        variant_ids.add(variant_id)
        transformed_groups.add(str(parent["group_id"]))

    if not generated:
        raise ValueError("No reviewed shift variants were generated")
    rebuilt_base = [
        _rebuild_with_split(row, "test")
        if str(row["group_id"]) in transformed_groups
        else row
        for row in parsed_base
    ]
    combined = [*rebuilt_base, *generated]
    group_splits: dict[str, set[str]] = {}
    for row in combined:
        group_splits.setdefault(str(row["group_id"]), set()).add(row["protocol_split"])
    leaking = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking:
        raise ValueError(
            f"Reviewed shift merge leaks groups across splits: {leaking[:5]}"
        )
    return combined


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
        description="Merge independently reviewed paraphrase/obfuscation prompts"
    )
    parser.add_argument("--base_scenarios", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    registry, registry_sha256 = load_falsification_registry(Path(args.registry))
    variant_path = Path(args.variants)
    rows = merge_reviewed_variants(
        _read_jsonl(Path(args.base_scenarios)),
        _read_jsonl(variant_path),
        registry=registry,
        registry_sha256=registry_sha256,
        review_file_sha256=file_sha256(variant_path),
    )
    _write_jsonl(Path(args.output), rows)
    print(f"saved {len(rows)} base and reviewed shifted scenarios to {args.output}")


if __name__ == "__main__":
    main()
