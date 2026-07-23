from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from cli.run_llm_judge_baselines import JudgeRuntime, _git_state, _load_judge_spec
from data.benign_audit import AUTO_SCREEN_SCHEMA_VERSION
from data.benign_screening import screened_text_sha256, validate_benign_rollout
from data.rollout_schema import canonical_json, content_hash


IMPLEMENTATION_FILES = (
    Path(__file__),
    Path(__file__).with_name("run_llm_judge_baselines.py"),
    Path(__file__).parents[1] / "data" / "llm_judge.py",
    Path(__file__).parents[1] / "data" / "benign_audit.py",
    Path(__file__).parents[1] / "data" / "benign_screening.py",
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


def _implementation_sha256() -> str:
    payload = {
        path.relative_to(Path(__file__).parents[1]).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in IMPLEMENTATION_FILES
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _interaction_text(rollout: dict[str, Any]) -> str:
    messages = [
        *rollout["prompt_messages"],
        {"role": "assistant", "content": rollout["response_text"]},
    ]
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )


def _release_runtime(runtime: JudgeRuntime) -> None:
    if hasattr(runtime, "model"):
        del runtime.model
    if hasattr(runtime, "tokenizer"):
        del runtime.tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except (ImportError, RuntimeError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run pinned distinct-family forced-choice screeners over blinded benign "
            "rollouts"
        )
    )
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--screener_config", required=True)
    parser.add_argument("--screener_model_keys", required=True)
    parser.add_argument("--monitored_family", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto|cpu|cuda|cuda:N|mps",
    )
    parser.add_argument("--allow_dirty_code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    model_keys = [
        value.strip() for value in args.screener_model_keys.split(",") if value.strip()
    ]
    if len(model_keys) < 3 or len(set(model_keys)) != len(model_keys):
        raise ValueError("Provide at least three distinct --screener_model_keys")

    rollout_path = Path(args.rollouts)
    rollouts = _read_jsonl(rollout_path)
    for rollout in rollouts:
        validate_benign_rollout(rollout)
    if len({row["rollout_id"] for row in rollouts}) != len(rollouts):
        raise ValueError("Rollout input contains duplicate rollout_id values")
    monitored_model_ids = {str(row.get("model_id", "")) for row in rollouts}
    if len(monitored_model_ids) != 1 or not next(iter(monitored_model_ids)):
        raise ValueError(
            "Automated screening requires one monitored model identity per file"
        )
    monitored_model_id = next(iter(monitored_model_ids))

    config_path = Path(args.screener_config)
    specs: list[tuple[str, dict[str, Any], str, str]] = []
    for model_key in model_keys:
        spec, config_sha256, spec_sha256 = _load_judge_spec(config_path, model_key)
        if spec.get("protocol_role") != "frozen_primary":
            raise ValueError(
                f"Benign screener {model_key!r} must have protocol_role=frozen_primary"
            )
        if spec["model_id"] == monitored_model_id:
            raise ValueError("The monitored model cannot screen its own responses")
        if str(spec["family"]).casefold() == args.monitored_family.strip().casefold():
            raise ValueError(
                f"Screener {model_key!r} is from the monitored model family"
            )
        threshold = spec.get("eligible_max_probability")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError(
                f"Screener {model_key!r} lacks a valid eligible_max_probability"
            )
        specs.append((model_key, spec, config_sha256, spec_sha256))
    families = [str(spec["family"]).casefold() for _, spec, _, _ in specs]
    if len(set(families)) != len(families):
        raise ValueError("Benign screeners must come from distinct model families")
    if len({config_hash for _, _, config_hash, _ in specs}) != 1:
        raise ValueError("Every screener must come from the same frozen config")

    code_commit, code_dirty = _git_state()
    if code_dirty and not args.allow_dirty_code:
        raise RuntimeError(
            "Refusing automated screening from a dirty worktree; commit the "
            "protocol or pass --allow_dirty_code for a non-final pilot"
        )
    implementation_sha256 = _implementation_sha256()
    rollout_file_sha256 = hashlib.sha256(rollout_path.read_bytes()).hexdigest()
    texts = np.asarray([_interaction_text(row) for row in rollouts], dtype=str)
    bundle = {
        "texts": texts,
        "labels": np.zeros(len(rollouts), dtype=np.int64),
        "example_ids": np.asarray([row["rollout_id"] for row in rollouts], dtype=str),
        "question_ids": np.asarray(
            [row.get("group_id", row["rollout_id"]) for row in rollouts],
            dtype=str,
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
        output.unlink()
    existing_rows = _read_jsonl(output) if output.exists() else []
    expected_contexts: dict[str, str] = {}
    screener_ids: dict[str, str] = {}
    for model_key, spec, config_sha256, spec_sha256 in specs:
        screener_id = content_hash(
            {
                "model_key": model_key,
                "model_id": spec["model_id"],
                "model_revision": spec["model_revision"],
                "tokenizer_revision": spec["tokenizer_revision"],
                "spec_sha256": spec_sha256,
            }
        )[:24]
        context_hash = content_hash(
            {
                "schema_version": AUTO_SCREEN_SCHEMA_VERSION,
                "rollout_file_sha256": rollout_file_sha256,
                "config_sha256": config_sha256,
                "spec_sha256": spec_sha256,
                "eligible_max_probability": float(spec["eligible_max_probability"]),
                "code_commit": code_commit,
                "implementation_sha256": implementation_sha256,
                "monitored_family": args.monitored_family,
            }
        )
        screener_ids[model_key] = screener_id
        expected_contexts[screener_id] = context_hash

    completed: set[tuple[str, str]] = set()
    valid_rollout_ids = {row["rollout_id"] for row in rollouts}
    for row in existing_rows:
        pair = (str(row.get("rollout_id", "")), str(row.get("screener_id", "")))
        if (
            pair[0] not in valid_rollout_ids
            or pair[1] not in expected_contexts
            or row.get("screening_context_sha256") != expected_contexts[pair[1]]
        ):
            raise ValueError(
                "Existing automated-screening output is stale or incompatible; "
                "use --overwrite to start a fresh run"
            )
        if pair in completed:
            raise ValueError("Existing automated-screening output has duplicates")
        completed.add(pair)

    generated = 0
    with output.open("a", encoding="utf-8") as handle:
        for model_key, spec, config_sha256, spec_sha256 in specs:
            screener_id = screener_ids[model_key]
            if all((row["rollout_id"], screener_id) in completed for row in rollouts):
                continue
            runtime = JudgeRuntime(
                spec,
                batch_size=args.batch_size,
                device=args.device,
            )
            scored = runtime.score_bundle(bundle, demonstrations=[])
            probabilities = scored["scores"].astype(float)
            threshold = float(spec["eligible_max_probability"])
            for index, rollout in enumerate(rollouts):
                pair = (rollout["rollout_id"], screener_id)
                if pair in completed:
                    continue
                probability = float(probabilities[index])
                record = {
                    "schema_version": AUTO_SCREEN_SCHEMA_VERSION,
                    "rollout_id": rollout["rollout_id"],
                    "screened_text_sha256": screened_text_sha256(rollout),
                    "screener_id": screener_id,
                    "screener_model_key": model_key,
                    "screener_model_id": spec["model_id"],
                    "screener_model_revision": spec["model_revision"],
                    "screener_tokenizer_revision": spec["tokenizer_revision"],
                    "screener_family": spec["family"],
                    "screener_spec_sha256": spec_sha256,
                    "config_sha256": config_sha256,
                    "eligible_max_probability": threshold,
                    "ineligible_probability": probability,
                    "decision": ("eligible" if probability <= threshold else "abstain"),
                    "judge_prompt_sha256": str(scored["prompt_sha256"][index]),
                    "judge_prompt_token_count": int(
                        scored["prompt_token_lengths"][index]
                    ),
                    "blinded_to_model_identity": True,
                    "blinded_to_monitor_scores": True,
                    "code_commit": code_commit,
                    "code_dirty": code_dirty,
                    "implementation_sha256": implementation_sha256,
                    "rollout_file_sha256": rollout_file_sha256,
                    "screening_context_sha256": expected_contexts[screener_id],
                }
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                completed.add(pair)
                generated += 1
            handle.flush()
            os.fsync(handle.fileno())
            _release_runtime(runtime)

    expected_count = len(rollouts) * len(specs)
    if len(completed) != expected_count:
        raise RuntimeError(
            f"Automated screening is incomplete: {len(completed)}/{expected_count}"
        )
    print(f"saved {expected_count} automated decisions ({generated} new) to {output}")


if __name__ == "__main__":
    main()
