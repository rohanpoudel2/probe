from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from data.benign_screening import screening_template


def build_screening_batch(
    rollouts: list[dict[str, Any]],
    *,
    batch_seed: int,
    annotator_id: str | None = None,
) -> list[dict[str, Any]]:
    if annotator_id is not None and not annotator_id.strip():
        raise ValueError("annotator_id must be non-empty when supplied")
    templates = [screening_template(row, batch_seed=batch_seed) for row in rollouts]
    screening_ids = [row["screening_id"] for row in templates]
    if len(set(screening_ids)) != len(screening_ids):
        raise ValueError("Screening batch contains duplicate content IDs")
    if annotator_id is not None:
        for row in templates:
            row["annotator_id"] = annotator_id
    templates.sort(key=lambda row: row["batch_order_key"])
    return templates


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
    parser.add_argument(
        "--rater_output",
        action="append",
        nargs=3,
        metavar=("ANNOTATOR_ID", "BATCH_SEED", "PATH"),
        help=(
            "Create an assigned, independently ordered rater copy. Repeat exactly "
            "once per rater; ANNOTATOR_ID, BATCH_SEED, and PATH must all be distinct."
        ),
    )
    args = parser.parse_args()

    rollouts = _read_jsonl(Path(args.rollouts))
    master_output = Path(args.output)
    rater_specs: list[tuple[str, int, Path]] = []
    for annotator_id, seed_text, path_text in args.rater_output or []:
        try:
            seed = int(seed_text)
        except ValueError as err:
            raise ValueError(
                f"Rater batch seed must be an integer, got {seed_text!r}"
            ) from err
        rater_specs.append((annotator_id, seed, Path(path_text)))

    if len(rater_specs) == 1:
        raise ValueError(
            "Benign screening requires at least two independent rater outputs"
        )
    annotator_ids = [spec[0] for spec in rater_specs]
    seeds = [spec[1] for spec in rater_specs]
    output_paths = [spec[2].resolve() for spec in rater_specs]
    if len(set(annotator_ids)) != len(annotator_ids):
        raise ValueError("Rater annotator IDs must be distinct")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Rater batch seeds must be distinct")
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Rater output paths must be distinct")
    if master_output.resolve() in output_paths:
        raise ValueError("The master template and rater output paths must be distinct")

    master = build_screening_batch(rollouts, batch_seed=args.batch_seed)
    rater_batches = [
        (
            output_path,
            build_screening_batch(
                rollouts,
                batch_seed=seed,
                annotator_id=annotator_id,
            ),
        )
        for annotator_id, seed, output_path in rater_specs
    ]

    _write_jsonl(master_output, master)
    print(f"saved {len(master)} blinded screening records to {master_output}")
    for output_path, batch in rater_batches:
        _write_jsonl(output_path, batch)
        print(f"saved {len(batch)} assigned blinded screening records to {output_path}")


if __name__ == "__main__":
    main()
