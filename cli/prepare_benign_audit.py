from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cli.prepare_benign_screening import build_screening_batch
from data.benign_audit import (
    bind_audit_rater_assignments,
    build_audit_manifest,
    build_automated_consensus,
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
            "Freeze an automated benign consensus and create a blinded human audit"
        )
    )
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--automated_decisions", required=True)
    parser.add_argument("--monitored_family", required=True)
    parser.add_argument("--min_screeners", type=int, default=3)
    parser.add_argument("--random_audit_size", type=int, default=300)
    parser.add_argument("--risk_audit_size", type=int, default=0)
    parser.add_argument("--selection_seed", type=int, default=9173)
    parser.add_argument("--master_batch_seed", type=int, default=104723)
    parser.add_argument("--output_consensus", required=True)
    parser.add_argument("--output_consensus_report", required=True)
    parser.add_argument("--output_audit_manifest", required=True)
    parser.add_argument("--output_template", required=True)
    parser.add_argument(
        "--rater_output",
        action="append",
        nargs=3,
        metavar=("ANNOTATOR_ID", "BATCH_SEED", "PATH"),
        required=True,
    )
    args = parser.parse_args()

    if len(args.rater_output) < 2:
        raise ValueError("Human audit requires at least two assigned rater outputs")
    rater_specs: list[tuple[str, int, Path]] = []
    for annotator_id, seed_text, path_text in args.rater_output:
        try:
            seed = int(seed_text)
        except ValueError as err:
            raise ValueError(
                f"Rater batch seed must be an integer, got {seed_text!r}"
            ) from err
        if not annotator_id.strip():
            raise ValueError("Rater annotator IDs must be non-empty")
        rater_specs.append((annotator_id, seed, Path(path_text)))
    annotator_ids = [item[0] for item in rater_specs]
    seeds = [item[1] for item in rater_specs]
    paths = [item[2].resolve() for item in rater_specs]
    if len(set(annotator_ids)) != len(annotator_ids):
        raise ValueError("Rater annotator IDs must be distinct")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Rater batch seeds must be distinct")
    if len(set(paths)) != len(paths):
        raise ValueError("Rater output paths must be distinct")

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
    manifest = build_audit_manifest(
        rollouts,
        consensus,
        random_sample_size=args.random_audit_size,
        risk_sample_size=args.risk_audit_size,
        selection_seed=args.selection_seed,
    )
    selected_ids = {record["rollout_id"] for record in manifest["records"]}
    selected_rollouts = [
        rollout for rollout in rollouts if rollout["rollout_id"] in selected_ids
    ]
    master = build_screening_batch(
        selected_rollouts,
        batch_seed=args.master_batch_seed,
    )
    rater_batches = [
        (
            output_path,
            build_screening_batch(
                selected_rollouts,
                batch_seed=seed,
                annotator_id=annotator_id,
            ),
        )
        for annotator_id, seed, output_path in rater_specs
    ]
    manifest = bind_audit_rater_assignments(
        manifest,
        [
            (annotator_id, seed, batch)
            for (annotator_id, seed, _), (_, batch) in zip(
                rater_specs,
                rater_batches,
                strict=True,
            )
        ],
    )

    _atomic_jsonl(Path(args.output_consensus), consensus)
    _atomic_json(Path(args.output_consensus_report), consensus_report)
    _atomic_json(Path(args.output_audit_manifest), manifest)
    _atomic_jsonl(Path(args.output_template), master)
    for output_path, batch in rater_batches:
        _atomic_jsonl(output_path, batch)
    print(
        f"saved {len(master)} blinded audit records "
        f"({manifest['n_random_auto_eligible']} random, "
        f"{manifest['n_risk_diagnostic']} risk diagnostic) for "
        f"{len(rater_batches)} independent raters"
    )


if __name__ == "__main__":
    main()
