from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from cli.common import load_yaml
from data.rollout_schema import ScenarioRecord


PREFILTER_VERSION = "wildchat-benign-candidate-v1"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
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
            yield line_number, row


def _normalized_prompt(text: str) -> str:
    return " ".join(text.casefold().split())


def _moderation_flagged(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("flagged") is True:
            return True
        return any(_moderation_flagged(item) for item in value.values())
    if isinstance(value, list):
        return any(_moderation_flagged(item) for item in value)
    return False


def _first_user_message(row: dict[str, Any]) -> str | None:
    conversation = row.get("conversation")
    if not isinstance(conversation, list):
        return None
    for message in conversation:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).casefold()
        content = message.get("content")
        if role in {"user", "human"} and isinstance(content, str) and content.strip():
            return content.strip()
    return None


def build_benign_scenarios(
    rows: Iterable[tuple[int, dict[str, Any]]],
    *,
    source_repo: str,
    source_revision: str,
    source_split: str,
    raw_file_sha256: str,
    max_scenarios: int,
    min_chars: int,
    max_chars: int,
    selection_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Create prompt-only candidates; this function never assigns benign labels."""

    if not COMMIT_RE.fullmatch(source_revision):
        raise ValueError(
            "Benign source revision must be a full 40-character commit hash"
        )
    if max_scenarios < 1 or min_chars < 1 or max_chars < min_chars:
        raise ValueError("Invalid candidate-count or character-length bounds")

    candidates: list[tuple[str, ScenarioRecord]] = []
    rejection_counts: Counter[str] = Counter()
    seen_prompts: set[str] = set()
    for line_number, row in rows:
        language = str(row.get("language", "")).strip().casefold()
        if language not in {"english", "en"}:
            rejection_counts["non_english"] += 1
            continue
        if row.get("toxic") is not False:
            rejection_counts["toxic_or_unknown"] += 1
            continue
        if row.get("redacted") is not False:
            rejection_counts["redacted_or_unknown"] += 1
            continue
        if _moderation_flagged(row.get("openai_moderation")):
            rejection_counts["moderation_flagged"] += 1
            continue
        prompt = _first_user_message(row)
        if prompt is None:
            rejection_counts["missing_user_prompt"] += 1
            continue
        if not min_chars <= len(prompt) <= max_chars:
            rejection_counts["length_out_of_bounds"] += 1
            continue
        normalized = _normalized_prompt(prompt)
        if normalized in seen_prompts:
            rejection_counts["duplicate_prompt"] += 1
            continue
        seen_prompts.add(normalized)

        prompt_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        conversation_hash = str(row.get("conversation_hash") or prompt_hash)
        rank = hashlib.sha256(
            f"{selection_seed}:{conversation_hash}:{prompt_hash}".encode("utf-8")
        ).hexdigest()
        group_id = f"benign_{prompt_hash[:20]}"
        scenario = ScenarioRecord(
            scenario_id=f"{group_id}__wildchat__{rank[:12]}",
            group_id=group_id,
            task_family="benign_calibration",
            messages=[{"role": "user", "content": prompt}],
            condition="natural_benign_candidate",
            protocol_split="calibration",
            source=f"{source_repo}@{source_revision}:{source_split}",
            metadata={
                "conversation_hash": conversation_hash,
                "language": row.get("language"),
                "source_line": line_number,
                "source_revision": source_revision,
                "raw_file_sha256": raw_file_sha256,
                "prefilter_version": PREFILTER_VERSION,
                "screening_status": "pending_independent_review",
                "upstream_toxic": False,
                "upstream_redacted": False,
            },
        )
        candidates.append((rank, scenario))

    selected = sorted(candidates, key=lambda item: item[0])[:max_scenarios]
    rejection_counts["eligible_before_sampling"] = len(candidates)
    rejection_counts["selected"] = len(selected)
    return [scenario.to_dict() for _, scenario in selected], dict(rejection_counts)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No benign candidates survived the prefilter")
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
        description="Build prompt-only natural-traffic candidates for independent benign screening"
    )
    parser.add_argument("--raw_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source_lock", default="experiments/data/huggingface_source_lock.yaml"
    )
    parser.add_argument("--max_scenarios", type=int, default=25_000)
    parser.add_argument("--min_chars", type=int, default=10)
    parser.add_argument("--max_chars", type=int, default=4_000)
    parser.add_argument("--selection_seed", type=int, default=42)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    raw_path = Path(args.raw_data)
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    source = load_yaml(args.source_lock)["sources"]["benign_calibration_raw"]["dataset"]
    scenarios, counts = build_benign_scenarios(
        _read_jsonl(raw_path),
        source_repo=source["repo"],
        source_revision=source["revision"],
        source_split=source.get("split", "train"),
        raw_file_sha256=raw_sha256,
        max_scenarios=args.max_scenarios,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        selection_seed=args.selection_seed,
    )
    _atomic_write_jsonl(Path(args.output), scenarios)
    report = {
        "status": "pass",
        "prefilter_version": PREFILTER_VERSION,
        "source": f"{source['repo']}@{source['revision']}",
        "raw_file_sha256": raw_sha256,
        "counts": counts,
        "warning": (
            "Upstream moderation is only a prefilter. These candidates are not benign labels; "
            "generate on-policy responses and complete independent screening."
        ),
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
