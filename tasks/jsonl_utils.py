from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


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
