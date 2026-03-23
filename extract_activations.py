"""CLI entry point for activation extraction (GPU phase).

Run this once per (model, dataset, mode) combination. The output is a set of
.npy files that all subsequent probe training reads from.

Usage:
    # Standard extraction (for P1-P4, P7)
    python extract_activations.py --config config.yaml --model Qwen/Qwen3-4B --dataset enron

    # Prompted extraction (for P6)
    python extract_activations.py --config config.yaml --model Qwen/Qwen3-4B --dataset enron --mode prompted

    # Follow-up extraction (for P8)
    python extract_activations.py --config config.yaml --model Qwen/Qwen3-4B --dataset enron --mode followup

    # Extract all modes for a model+dataset
    python extract_activations.py --config config.yaml --model Qwen/Qwen3-4B --dataset enron --mode all

    # Skip filtering (use all samples)
    python extract_activations.py --config config.yaml --model Qwen/Qwen3-4B --dataset enron --skip-filtering
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from data.loading import load_enron, load_sms
from data.filtering import filter_by_logit_confidence, save_filtered_indices
from data.splitting import make_splits, save_splits
from extraction.extractor import extract_and_cache, _sanitize_model_name, get_device, get_dtype, empty_cache
from extraction.modified_extractor import extract_prompted, extract_followup


DATASET_LOADERS = {
    "enron": load_enron,
    "sms": load_sms,
}


def run_extraction(config: dict, model_name: str, dataset_name: str,
                   mode: str = "standard", skip_filtering: bool = False):
    """Run activation extraction for one model+dataset+mode combination."""

    # Find model config
    model_cfg = None
    for m in config["models"]:
        if m["name"] == model_name:
            model_cfg = m
            break
    if model_cfg is None:
        raise ValueError(f"Model {model_name} not found in config")

    layers = model_cfg["layers"]
    cache_dir = config["extraction"]["cache_dir"]
    batch_size = config["extraction"]["batch_size"]
    max_length = config["extraction"]["max_length"]

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    loader = DATASET_LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    dataset = loader()
    print(f"  {len(dataset)} samples loaded")

    texts = dataset["text"]
    labels = np.array(dataset["label"])

    # Filtering (optional, requires GPU)
    cache_path = Path(cache_dir) / _sanitize_model_name(model_name) / dataset_name
    filter_indices_path = cache_path / "filtered_indices.json"

    if not skip_filtering and not filter_indices_path.exists():
        print("Running logit-difference filtering (requires GPU/MPS)...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = get_device()
        dtype = get_dtype(device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if device.type == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, device_map="auto"
            )

        threshold = config["filtering"]["logit_diff_threshold"]
        dataset, _ = filter_by_logit_confidence(
            dataset, model, tokenizer, threshold=threshold,
            batch_size=batch_size, max_length=max_length,
        )
        texts = dataset["text"]
        labels = np.array(dataset["label"])

        # Save filtered indices
        save_filtered_indices(np.arange(len(labels)), filter_indices_path)

        # Free the model — we'll reload it in extraction with hooks
        del model
        empty_cache()

        print(f"  {len(texts)} samples after filtering")
    else:
        if skip_filtering:
            print("  Skipping filtering (--skip-filtering)")
        else:
            print(f"  Using existing filtered indices from {filter_indices_path}")

    # Create splits (only for training datasets, not OOD test)
    splits_path = cache_path / "splits.json"
    if not splits_path.exists():
        print("Creating reproducible splits...")
        splits = make_splits(dataset)
        cache_path.mkdir(parents=True, exist_ok=True)
        save_splits(splits, splits_path)
        pos = (labels == 1).sum()
        neg = (labels == 0).sum()
        print(f"  {pos} positives, {neg} negatives")
        print(f"  Train pool: {len(splits['train_pool'])}, "
              f"Eval: {len(splits['eval'])}, Test: {len(splits['test'])}")

    # Run extraction
    modes_to_run = ["standard", "prompted", "followup"] if mode == "all" else [mode]

    for m in modes_to_run:
        print(f"\nExtracting activations (mode={m})...")
        if m == "standard":
            extract_and_cache(
                model_name=model_name,
                texts=texts,
                labels=labels,
                layers=layers,
                cache_dir=cache_dir,
                dataset_name=dataset_name,
                batch_size=batch_size,
                max_length=max_length,
            )
        elif m == "prompted":
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
        elif m == "followup":
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
            raise ValueError(f"Unknown mode: {m}")


def main():
    parser = argparse.ArgumentParser(description="Extract and cache model activations")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen3-4B)")
    parser.add_argument("--dataset", required=True, help="Dataset name (enron or sms)")
    parser.add_argument("--mode", default="standard",
                        choices=["standard", "prompted", "followup", "all"],
                        help="Extraction mode")
    parser.add_argument("--skip-filtering", action="store_true",
                        help="Skip logit-difference filtering")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_extraction(config, args.model, args.dataset, args.mode, args.skip_filtering)


if __name__ == "__main__":
    main()
