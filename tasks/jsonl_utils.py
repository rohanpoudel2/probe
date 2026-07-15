from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def require_fields(row: Dict, fields: Iterable[str], path: str) -> None:
    missing = [f for f in fields if f not in row]
    if missing:
        raise ValueError(f"Missing required fields {missing} in {path}: {row}")


def rollout_metadata(row: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """Keep the provenance fields needed by extraction and black-box baselines."""

    scenario_metadata = row.get("metadata", {})
    if not isinstance(scenario_metadata, dict):
        raise ValueError("Rollout scenario metadata must be an object")
    return {
        "source": row.get("source", "jsonl"),
        "data_origin": row.get("data_origin"),
        "generated_by_model": row.get("generated_by_model"),
        "rollout_id": row.get("rollout_id"),
        "scenario_id": row.get("scenario_id"),
        "protocol_split": row.get("protocol_split"),
        "model_id": row.get("model_id"),
        "model_revision": row.get("model_revision"),
        "tokenizer_revision": row.get("tokenizer_revision"),
        "label_source": row.get("label_source"),
        "annotation_protocol": row.get("annotation_protocol"),
        "annotation_metadata": row.get("annotation_metadata", {}),
        "generation": row.get("generation"),
        "eligible_for_main_study": row.get("eligible_for_main_study"),
        "construct_name": row.get("construct_name"),
        "scenario_metadata": scenario_metadata,
        "falsification": scenario_metadata.get("falsification"),
        **extra,
    }
