"""CLI entry point for activation extraction (GPU phase)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from data.filtering import (
    apply_saved_filter,
    filter_by_logit_confidence,
    load_filtered_indices,
    save_filtered_indices,
)
from data.loading import DATASET_SPECS, load_enron, load_mask, load_sms
from data.splitting import make_splits, save_split_manifest, save_splits
from extraction.extractor import _sanitize_model_name, extract_and_cache
from extraction.modified_extractor import extract_followup, extract_prompted


DATASET_LOADERS = {
    "enron": load_enron,
    "sms": load_sms,
    "mask": load_mask,
}


def _get_prompting_config(config: dict, dataset_name: str) -> dict:
    return config.get("prompting", {}).get(dataset_name, {})

def _load_dataset(dataset_name: str, config: dict):
    loader = DATASET_LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_name == "mask":
        return loader()
    return loader()


def _write_manifest(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_extraction(
    config: dict,
    model_name: str,
    dataset_name: str,
    mode: str = "standard",
    skip_filtering: bool = False,
    overwrite: bool = False,
):
    """Run activation extraction for one model/dataset/mode combination."""
    model_cfg = next((m for m in config["models"] if m["name"] == model_name), None)
    if model_cfg is None:
        raise ValueError(f"Model {model_name} not found in config")

    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    layers = model_cfg["layers"]
    cache_dir = config["extraction"]["cache_dir"]
    batch_size = config["extraction"]["batch_size"]
    max_length = config["extraction"]["max_length"]

    print(f"Loading dataset: {dataset_name}")
    original_dataset = _load_dataset(dataset_name, config)
    print(f"  {len(original_dataset)} samples loaded")

    dataset = original_dataset
    cache_path = Path(cache_dir) / _sanitize_model_name(model_name) / dataset_name
    filter_indices_path = cache_path / "filtered_indices.json"

    if skip_filtering:
        print("  Skipping filtering (--skip-filtering)")
        kept_indices = np.arange(len(dataset))
        filtered_dataset = dataset
    elif filter_indices_path.exists() and not overwrite:
        print(f"  Reusing filtered indices from {filter_indices_path}")
        kept_indices = load_filtered_indices(filter_indices_path)
        filtered_dataset = apply_saved_filter(dataset, kept_indices)
    else:
        print("Running logit-difference filtering (requires GPU)...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        threshold = config["filtering"]["logit_diff_threshold"]
        filtered_dataset, kept_indices, kept_logit_diffs = filter_by_logit_confidence(
            dataset,
            model,
            tokenizer,
            dataset_name=dataset_name,
            threshold=threshold,
            batch_size=batch_size,
            max_length=max_length,
        )
        save_filtered_indices(kept_indices, filter_indices_path)

        del model
        torch.cuda.empty_cache()

        print(f"  {len(filtered_dataset)} samples after filtering")
        _write_manifest(
            cache_path / "filter_manifest.json",
            {
                "dataset": dataset_name,
                "model": model_name,
                "threshold": threshold,
                "n_original": len(original_dataset),
                "n_filtered": len(filtered_dataset),
                "kept_indices_path": str(filter_indices_path),
                "mean_kept_logit_diff": float(np.mean(kept_logit_diffs)) if len(kept_logit_diffs) else None,
            },
        )

    dataset = filtered_dataset
    texts = dataset["text"]
    labels = np.array(dataset["label"])

    base_manifest = {
        "model": model_name,
        "dataset": dataset_name,
        "n_original": len(original_dataset),
        "n_filtered": len(dataset),
        "filter_applied": not skip_filtering,
        "filtered_indices_path": str(filter_indices_path),
        "layers": layers,
        "max_length": max_length,
        "batch_size": batch_size,
        "overwrite": overwrite,
    }
    _write_manifest(cache_path / "manifest_standard.json", {**base_manifest, "mode": "standard"})

    splits_path = cache_path / "splits.json"
    split_manifest_path = cache_path / "splits_manifest.json"
    if not splits_path.exists() or overwrite:
        print("Creating reproducible splits...")
        splits = make_splits(dataset)
        cache_path.mkdir(parents=True, exist_ok=True)
        save_splits(splits, splits_path)
        save_split_manifest(dataset_name, splits, labels, split_manifest_path)
        pos = int((labels == 1).sum())
        neg = int((labels == 0).sum())
        print(f"  {pos} positives, {neg} negatives")
        print(
            f"  Train pool: {len(splits['train_pool'])}, "
            f"Eval: {len(splits['eval'])}, Test: {len(splits['test'])}"
        )

    modes_to_run = ["standard", "prompted", "followup"] if mode == "all" else [mode]

    for current_mode in modes_to_run:
        print(f"\nExtracting activations (mode={current_mode})...")
        manifest = {**base_manifest, "mode": current_mode}
        _write_manifest(cache_path / f"manifest_{current_mode}.json", manifest)

        if current_mode == "standard":
            extract_and_cache(
                model_name=model_name,
                texts=texts,
                labels=labels,
                layers=layers,
                cache_dir=cache_dir,
                dataset_name=dataset_name,
                batch_size=batch_size,
                max_length=max_length,
                overwrite=overwrite,
            )
        elif current_mode == "prompted":
            extract_prompted(
                model_name=model_name,
                texts=texts,
                labels=labels,
                layers=layers,
                cache_dir=cache_dir,
                dataset_name=dataset_name,
                batch_size=batch_size,
                max_length=max_length,
            )
        elif current_mode == "followup":
            extract_followup(
                model_name=model_name,
                texts=texts,
                labels=labels,
                layers=layers,
                cache_dir=cache_dir,
                dataset_name=dataset_name,
                batch_size=batch_size,
                max_length=max_length,
            )
        else:
            raise ValueError(f"Unknown mode: {current_mode}")


def main():
    parser = argparse.ArgumentParser(description="Extract and cache model activations")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen3-4B)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., enron, sms, mask)")
    parser.add_argument(
        "--mode",
        default="standard",
        choices=["standard", "prompted", "followup", "all"],
        help="Extraction mode",
    )
    parser.add_argument("--skip-filtering", action="store_true", help="Skip logit-difference filtering")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate filtered indices and cached activations")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_extraction(
        config,
        args.model,
        args.dataset,
        args.mode,
        args.skip_filtering,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
