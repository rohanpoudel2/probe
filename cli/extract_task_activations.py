from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import yaml

from data.group_splitting import grouped_train_eval_test_split
from extraction.task_extractor import TaskActivationExtractor, TaskExtractionConfig
from tasks import TASK_REGISTRY


def _parse_layers(raw_layers) -> List[int]:
    if isinstance(raw_layers, list):
        return [int(x) for x in raw_layers]
    return [int(x.strip()) for x in str(raw_layers).split(",") if x.strip()]


def _default_prompt_prefix(task_name: str) -> str:
    return (
        "You are producing internal monitoring features for the behavior task "
        f"'{task_name}'. Focus on whether the response shows socially misaligned reasoning."
    )


def _mode_kwargs(task_name: str, mode: str) -> dict:
    if mode == "standard":
        return {"modified_mode": "standard", "prompt_prefix": None, "prompt_suffix": None}
    if mode == "prompted":
        return {
            "modified_mode": "prompted",
            "prompt_prefix": _default_prompt_prefix(task_name),
            "prompt_suffix": None,
        }
    raise ValueError(f"Unknown modified mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Span-aware extraction for structured task families")
    parser.add_argument("--task", required=True, choices=sorted(TASK_REGISTRY.keys()))
    parser.add_argument("--data", required=True, help="JSONL path for the selected task")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices")
    parser.add_argument("--views", default="full_text,answer", help="Comma-separated named spans/views")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--pooling_mode", default="mean", choices=["mean", "last"])
    parser.add_argument("--modified_modes", default="standard", help="Comma-separated extraction modes: standard,prompted")
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--eval_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task = TASK_REGISTRY[args.task]()
    examples = task.load(args.data)
    splits = grouped_train_eval_test_split(
        examples,
        group_key=task.spec.grouped_split_key,
        train_frac=args.train_frac,
        eval_frac=args.eval_frac,
        seed=args.seed,
    )

    modes = [mode.strip() for mode in args.modified_modes.split(",") if mode.strip()]
    for mode in modes:
        extractor = TaskActivationExtractor(
            TaskExtractionConfig(
                model_name=args.model,
                layers=_parse_layers(args.layers),
                max_length=args.max_length,
                pooling_mode=args.pooling_mode,
                views=[v.strip() for v in args.views.split(",") if v.strip()],
                output_dir=args.output_dir,
                **_mode_kwargs(args.task, mode),
            )
        )

        for split_name, split_examples in splits.items():
            if split_examples:
                extractor.extract_split(split_examples, split_name)
                print(f"saved {split_name} [{mode}] -> {len(split_examples)} examples")


if __name__ == "__main__":
    main()
