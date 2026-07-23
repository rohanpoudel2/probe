from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from data.benign_audit import (
    build_automated_consensus,
    merge_audited_consensus,
)


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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the human audit and materialize conservative benign labels"
        )
    )
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--automated_decisions", required=True)
    parser.add_argument("--audit_manifest", required=True)
    parser.add_argument("--ratings", required=True, nargs="+")
    parser.add_argument("--monitored_family", required=True)
    parser.add_argument("--min_screeners", type=int, default=3)
    parser.add_argument("--min_independent_raters", type=int, default=2)
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument(
        "--max_false_acceptance_rate",
        type=float,
        default=0.01,
    )
    parser.add_argument("--output_annotations", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    rollouts = _read_jsonl(Path(args.rollouts))
    decisions = _read_jsonl(Path(args.automated_decisions))
    consensus, consensus_report = build_automated_consensus(
        rollouts,
        decisions,
        monitored_family=args.monitored_family,
        min_screeners=args.min_screeners,
    )
    if consensus_report["status"] != "pass":
        raise ValueError("No rollout passed the frozen automated consensus")
    manifest = json.loads(Path(args.audit_manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Audit manifest must be a JSON object")
    ratings = [
        row for rating_path in args.ratings for row in _read_jsonl(Path(rating_path))
    ]
    annotations, report = merge_audited_consensus(
        rollouts,
        consensus,
        manifest,
        ratings,
        confidence_level=args.confidence_level,
        max_false_acceptance_rate=args.max_false_acceptance_rate,
        min_independent_raters=args.min_independent_raters,
    )
    _atomic_json(Path(args.report), report)
    if report["status"] != "pass":
        raise ValueError(
            "Human audit did not validate the automated benign screening pipeline; "
            "annotations were not written"
        )
    _atomic_jsonl(Path(args.output_annotations), annotations)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
