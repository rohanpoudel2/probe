from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from data.rollout_schema import RolloutRecord, ScenarioRecord, content_hash
from data.generation_confidence import build_generation_confidence_trace


def _read_scenarios(path: Path) -> list[ScenarioRecord]:
    scenarios: list[ScenarioRecord] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                scenario = ScenarioRecord.from_dict(json.loads(line))
            except Exception as err:
                raise ValueError(
                    f"Invalid scenario at {path}:{line_number}: {err}"
                ) from err
            if scenario.scenario_id in seen:
                raise ValueError(f"Duplicate scenario_id {scenario.scenario_id!r}")
            seen.add(scenario.scenario_id)
            scenarios.append(scenario)
    if not scenarios:
        raise ValueError(f"No scenarios found in {path}")
    return scenarios


def _existing_rollout_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rollout_id = row.get("rollout_id")
            if not rollout_id:
                raise ValueError(
                    f"Existing output lacks rollout_id at {path}:{line_number}"
                )
            ids.add(str(rollout_id))
    return ids


def _git_provenance() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"code_commit": commit, "code_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"code_commit": "unknown", "code_dirty": None}


def _split_reasoning(response: str) -> tuple[str | None, str]:
    stripped = response.strip()
    if "<think>" in stripped and "</think>" in stripped:
        before, remainder = stripped.split("<think>", 1)
        reasoning, after = remainder.split("</think>", 1)
        final_answer = (before + after).strip()
        return reasoning.strip() or None, final_answer or stripped
    return None, stripped


def _rollout_id(
    scenario: ScenarioRecord,
    model_id: str,
    model_revision: str,
    replicate: int,
    seed: int,
) -> str:
    digest = content_hash(
        {
            "scenario_id": scenario.scenario_id,
            "scenario_hash": scenario.to_dict()["scenario_hash"],
            "model_id": model_id,
            "model_revision": model_revision,
            "replicate": replicate,
            "seed": seed,
        }
    )[:20]
    return f"{scenario.scenario_id}__r{replicate}__{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate resumable on-policy task rollouts"
    )
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_revision", required=True)
    parser.add_argument("--tokenizer_revision", default=None)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto|cpu|cuda|cuda:N|mps",
    )
    parser.add_argument(
        "--allow_dirty_code",
        action="store_true",
        help="Permit generation from a dirty worktree for an explicitly non-final pilot.",
    )
    parser.add_argument(
        "--no_thinking",
        action="store_true",
        help="Disable the model's reasoning trace (enable_thinking=False) for a faster "
             "answer-view pilot. Non-final; the reasoning view is unavailable in this mode.",
    )
    args = parser.parse_args()

    if args.model_revision in {"main", "latest", "unpinned"}:
        raise ValueError("--model_revision must be immutable, not main/latest/unpinned")
    if args.num_rollouts < 1 or args.max_new_tokens < 1:
        raise ValueError("num_rollouts and max_new_tokens must be positive")
    if args.temperature < 0.0 or not 0.0 < args.top_p <= 1.0:
        raise ValueError("temperature must be non-negative and top_p must be in (0, 1]")
    if args.num_rollouts > 1 and args.temperature == 0.0:
        raise ValueError(
            "Multiple deterministic rollouts would duplicate one greedy completion; "
            "use num_rollouts=1 or register stochastic temperature/top_p sampling"
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from cli.common import resolve_torch_device

    scenario_path = Path(args.scenarios)
    scenarios = _read_scenarios(scenario_path)
    scenario_file_sha256 = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
        output.unlink()
    completed = _existing_rollout_ids(output)

    git = _git_provenance()
    if git["code_commit"] == "unknown":
        raise RuntimeError("Rollout generation requires Git provenance")
    if git["code_dirty"] and not args.allow_dirty_code:
        raise RuntimeError(
            "Refusing generation from a dirty worktree; commit the protocol or pass --allow_dirty_code for a non-final pilot"
        )

    tokenizer_revision = args.tokenizer_revision or args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=tokenizer_revision)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"Tokenizer for {args.model} has no chat template")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    resolved_device = resolve_torch_device(args.device)
    model_kwargs = {"revision": args.model_revision, "torch_dtype": "auto"}
    if resolved_device == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )
    if resolved_device != "auto":
        model = model.to(resolved_device)
    model.eval()
    model_device = next(model.parameters()).device
    chat_template_hash = content_hash(tokenizer.chat_template)
    generation_spec = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    generated_count = 0
    with output.open("a", encoding="utf-8") as handle:
        for scenario_index, scenario in enumerate(scenarios):
            # The prompt encoding depends only on the scenario, not the replicate
            # or seed, so build it once per scenario. Sampling randomness is set
            # by set_seed() immediately before each generate() call below, and
            # generate() does not mutate its inputs, so replicates remain
            # independent and their outputs are unchanged.
            template_kwargs = {"enable_thinking": False} if args.no_thinking else {}
            encoded = tokenizer.apply_chat_template(
                scenario.messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **template_kwargs,
            )
            if isinstance(encoded, torch.Tensor):
                encoded = {
                    "input_ids": encoded,
                    "attention_mask": torch.ones_like(encoded),
                }
            encoded = {
                key: value.to(model_device) for key, value in encoded.items()
            }
            for replicate in range(args.num_rollouts):
                rollout_seed = (
                    args.seed + scenario_index * args.num_rollouts + replicate
                )
                rollout_id = _rollout_id(
                    scenario, args.model, args.model_revision, replicate, rollout_seed
                )
                if rollout_id in completed:
                    continue
                set_seed(rollout_seed)
                kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.temperature > 0.0,
                    "pad_token_id": tokenizer.pad_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
                if args.temperature > 0.0:
                    kwargs.update(
                        {"temperature": args.temperature, "top_p": args.top_p}
                    )
                with torch.no_grad():
                    generated = model.generate(**encoded, **kwargs)
                prompt_length = int(encoded["input_ids"].shape[1])
                response_ids = generated.sequences[0, prompt_length:]
                confidence_trace = build_generation_confidence_trace(
                    generated.scores, response_ids
                )
                response = tokenizer.decode(
                    response_ids, skip_special_tokens=True
                ).strip()
                if not response:
                    raise RuntimeError(
                        f"Model produced an empty rollout for {scenario.scenario_id}"
                    )
                reasoning, final_answer = _split_reasoning(response)
                messages = [
                    *scenario.messages,
                    {"role": "assistant", "content": response},
                ]
                record = RolloutRecord(
                    rollout_id=rollout_id,
                    scenario=scenario,
                    response_text=response,
                    messages=messages,
                    model_id=args.model,
                    model_revision=args.model_revision,
                    tokenizer_revision=tokenizer_revision,
                    seed=rollout_seed,
                    generation={
                        **generation_spec,
                        "prompt_token_count": prompt_length,
                        "response_token_count": int(len(response_ids)),
                        "response_token_ids": response_ids.detach().cpu().tolist(),
                        "confidence_trace": confidence_trace,
                    },
                    provenance={
                        **git,
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "chat_template_sha256": chat_template_hash,
                        "scenario_file": str(scenario_path.resolve()),
                        "scenario_file_sha256": scenario_file_sha256,
                    },
                    reasoning=reasoning,
                    final_answer=final_answer,
                )
                handle.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
                completed.add(rollout_id)
                generated_count += 1

    print(
        f"generated {generated_count} new rollouts; {len(completed)} total in {output}"
    )


if __name__ == "__main__":
    main()
