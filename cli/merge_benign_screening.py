from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from data.benign_screening import merge_screening_ratings


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


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge independent benign-screening ratings into conservative annotations"
    )
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--ratings", required=True, nargs="+")
    parser.add_argument("--output_annotations", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min_independent_raters", type=int, default=2)
    args = parser.parse_args()

    ratings = [
        row
        for rating_path in args.ratings
        for row in _read_jsonl(Path(rating_path))
    ]
    annotations, report = merge_screening_ratings(
        _read_jsonl(Path(args.rollouts)),
        ratings,
        min_independent_raters=args.min_independent_raters,
    )
    if report["status"] != "pass":
        raise ValueError("No rollout passed independent benign screening")
    _atomic_jsonl(Path(args.output_annotations), annotations)
    _atomic_json(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
