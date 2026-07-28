from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import List

from data.group_splitting import declared_protocol_split
from extraction.task_extractor import TaskActivationExtractor, TaskExtractionConfig
from tasks import TASK_REGISTRY


def _parse_layers(raw_layers) -> List[int]:
    if isinstance(raw_layers, list):
        return [int(x) for x in raw_layers]
    return [int(x.strip()) for x in str(raw_layers).split(",") if x.strip()]


def _parse_percentiles(raw_percentiles) -> List[int] | None:
    if raw_percentiles is None:
        return None
    if isinstance(raw_percentiles, list):
        values = [int(x) for x in raw_percentiles]
    else:
        values = [int(x.strip()) for x in str(raw_percentiles).split(",") if x.strip()]
    return values or None


def _git_state() -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError) as err:
        raise RuntimeError("Activation extraction requires Git provenance") from err
    return revision, dirty


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Span-aware extraction for structured task families"
    )
    parser.add_argument("--task", required=True, choices=sorted(TASK_REGISTRY.keys()))
    parser.add_argument(
        "--data", required=True, help="JSONL path for the selected task"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model_revision", required=True, help="Pinned model commit or immutable tag"
    )
    parser.add_argument(
        "--tokenizer_revision", default=None, help="Defaults to --model_revision"
    )
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices")
    parser.add_argument(
        "--views", default="full_text,answer", help="Comma-separated named spans/views"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--allow_truncation",
        action="store_true",
        help=(
            "Permit explicit left truncation for a non-confirmatory run. "
            "The default is to fail if any rendered input exceeds max_length."
        ),
    )
    parser.add_argument("--pooling_mode", default="mean", choices=["mean", "last"])
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Use raw concatenation for a pre-registered base-model study.",
    )
    parser.add_argument(
        "--missing_view_policy", default="error", choices=["error", "drop"]
    )
    parser.add_argument(
        "--allow_non_model_generated_debug",
        action="store_true",
        help="Permit authored fixtures in an explicitly non-confirmatory debug run.",
    )
    parser.add_argument(
        "--trajectory_prefix_percentiles",
        default=None,
        help=(
            "Optional comma-separated percentiles (1-100) for trajectory-prefix "
            "views, e.g. '10,25,50,75,100'"
        ),
    )
    parser.add_argument(
        "--trajectory_prefix_stack_views",
        action="store_true",
        help=(
            "Emit trajectory_prefix_stack_p* views that concatenate cumulative prefix "
            "activations for sequence probes."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_dirty_code",
        action="store_true",
        help="Permit a dirty Git worktree for an explicitly non-confirmatory debug run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto|cpu|cuda|cuda:N|mps",
    )
    args = parser.parse_args()

    task = TASK_REGISTRY[args.task]()
    data_path = Path(args.data)
    dataset_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    code_revision, code_dirty = _git_state()
    if code_dirty and not args.allow_dirty_code:
        raise RuntimeError(
            "Refusing extraction from a dirty worktree; commit the protocol or "
            "pass --allow_dirty_code for a non-confirmatory debug run"
        )
    examples = task.load(args.data)
    splits = declared_protocol_split(
        examples, group_key=task.spec.grouped_split_key
    )

    extractor = TaskActivationExtractor(
        TaskExtractionConfig(
            model_name=args.model,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            layers=_parse_layers(args.layers),
            max_length=args.max_length,
            allow_truncation=args.allow_truncation,
            pooling_mode=args.pooling_mode,
            views=[v.strip() for v in args.views.split(",") if v.strip()],
            trajectory_prefix_percentiles=_parse_percentiles(
                args.trajectory_prefix_percentiles
            ),
            trajectory_prefix_stack_view=args.trajectory_prefix_stack_views,
            device=args.device,
            output_dir=args.output_dir,
            use_chat_template=not args.no_chat_template,
            missing_view_policy=args.missing_view_policy,
            require_model_generated=not args.allow_non_model_generated_debug,
            dataset_sha256=dataset_sha256,
            code_revision=code_revision,
            code_dirty=code_dirty,
            split_seed=args.seed,
        )
    )

    for split_name, split_examples in splits.items():
        if split_examples:
            extractor.extract_split(split_examples, split_name)
            print(f"saved {split_name} -> {len(split_examples)} examples")


if __name__ == "__main__":
    main()
