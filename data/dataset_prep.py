from __future__ import annotations

import json
from pathlib import Path
from string import Formatter
from typing import Any


def _flatten(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        full_key = f"{prefix}.{key}" if prefix else key
        flat[full_key] = value
        if isinstance(value, dict):
            flat.update(_flatten(value, full_key))
    return flat


def _get_field(row: dict[str, Any], path: str) -> Any:
    if path in row:
        return row[path]
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _first_nonempty(values: list[Any]) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def resolve_value(row: dict[str, Any], spec: Any) -> Any:
    if spec is None:
        return None
    if isinstance(spec, str):
        return _get_field(row, spec)
    if isinstance(spec, list):
        return _first_nonempty([resolve_value(row, item) for item in spec])
    if not isinstance(spec, dict):
        return spec

    if "const" in spec:
        value = spec["const"]
    elif "field" in spec:
        value = _get_field(row, spec["field"])
    elif "fields" in spec:
        value = _first_nonempty([resolve_value(row, item) for item in spec["fields"]])
    elif "template" in spec:
        flat = _flatten(row)
        mapping = {name: flat.get(name, "") for _, name, _, _ in Formatter().parse(spec["template"]) if name}
        value = spec["template"].format(**mapping)
    elif "parts" in spec:
        sep = spec.get("sep", "")
        parts = [resolve_value(row, item) for item in spec["parts"]]
        value = sep.join(str(part) for part in parts if part not in (None, ""))
    else:
        value = None

    if "map" in spec:
        mapping = spec["map"]
        if value in mapping:
            value = mapping[value]
        elif str(value) in mapping:
            value = mapping[str(value)]

    if value in (None, "") and "default" in spec:
        value = spec["default"]

    if spec.get("strip", True) and isinstance(value, str):
        value = value.strip()
    return value


def _passes_filters(row: dict[str, Any], filters: list[dict[str, Any]]) -> bool:
    for flt in filters:
        value = resolve_value(row, flt.get("field"))
        if "equals" in flt and value != flt["equals"]:
            return False
        if "not_equals" in flt and value == flt["not_equals"]:
            return False
        if "in" in flt and value not in set(flt["in"]):
            return False
        if "not_in" in flt and value in set(flt["not_in"]):
            return False
        if flt.get("truthy") and not value:
            return False
        if flt.get("falsy") and value:
            return False
    return True


def _normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Could not coerce label to int: {value!r}")


def normalize_rows(task_name: str, rows: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    fields = spec["fields"]
    required_fields = set(spec.get("required_fields", ["prompt", "label"]))
    filters = spec.get("filters", [])
    normalized: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        if not _passes_filters(row, filters):
            continue

        item = {out_key: resolve_value(row, field_spec) for out_key, field_spec in fields.items()}
        if not item.get("example_id"):
            item["example_id"] = f"{task_name}_{idx}"
        if not item.get("question_id"):
            item["question_id"] = item["example_id"]
        if "label" in item:
            item["label"] = _normalize_label(item["label"])
        if not item.get("source"):
            item["source"] = spec.get("source_name", "huggingface")

        missing = [name for name in required_fields if item.get(name) in (None, "")]
        if missing:
            raise ValueError(f"Missing normalized required fields {missing} for task={task_name}, row={idx}")
        normalized.append(item)
    return normalized


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

