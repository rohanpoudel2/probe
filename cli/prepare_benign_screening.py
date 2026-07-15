from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from data.benign_screening import screening_template


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
        description="Create a model-identity-blinded benign-screening batch"
    )
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_seed", type=int, default=42)
    args = parser.parse_args()

    templates = [
        screening_template(row, batch_seed=args.batch_seed)
        for row in _read_jsonl(Path(args.rollouts))
    ]
    screening_ids = [row["screening_id"] for row in templates]
    if len(set(screening_ids)) != len(screening_ids):
        raise ValueError("Screening batch contains duplicate content IDs")
    templates.sort(key=lambda row: row["batch_order_key"])
    _write_jsonl(Path(args.output), templates)
    print(f"saved {len(templates)} blinded screening records to {args.output}")


if __name__ == "__main__":
    main()
