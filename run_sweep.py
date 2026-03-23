"""Sweep orchestrator: run all (probe, k, seed, balance_mode, layer) experiments.

This is the CPU phase. It reads from the activation cache and never touches
the GPU. Each experiment takes milliseconds to seconds.

Usage:
    python run_sweep.py --config config.yaml
    python run_sweep.py --config config.yaml --model Qwen/Qwen3-4B --dataset enron
    python run_sweep.py --config config.yaml --probes P1_logistic P2_mass_mean
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from data.splitting import load_splits, sample_train_set
from extraction.extractor import load_cached_activations, _sanitize_model_name
from evaluation.metrics import compute_auroc, compute_recall_at_fpr
from probes import PROBE_REGISTRY


def _get_dataset_suffix(probe_cls):
    """Return the activation cache suffix for a probe type."""
    mod = probe_cls.requires_modified_activations
    if mod is None:
        return ""
    return f"_{mod}"


def run_single_experiment(
    probe_cls,
    activations: np.ndarray,
    labels: np.ndarray,
    splits: dict,
    k: int,
    seed: int,
    balance_mode: str,
    ood_activations: np.ndarray | None = None,
    ood_labels: np.ndarray | None = None,
) -> dict:
    """Run a single probe experiment and return metrics."""
    t0 = time.time()

    try:
        X_train, y_train = sample_train_set(
            activations, labels, splits["train_pool"], k, seed, balance_mode
        )
    except ValueError:
        # Not enough positives for this k
        return None

    probe = probe_cls()

    try:
        probe.fit(X_train, y_train)
    except Exception:
        # Some probes may fail at very low k (e.g., LDA with k=1)
        return {
            "auroc": float("nan"),
            "recall_at_1pct_fpr": float("nan"),
            "ood_auroc": float("nan"),
            "ood_recall_at_1pct_fpr": float("nan"),
            "wall_clock_s": time.time() - t0,
            "error": True,
        }

    # Evaluate on in-distribution test set
    test_idx = splits["test"]
    X_test = activations[test_idx]
    y_test = labels[test_idx]
    scores = probe.score(X_test)

    result = {
        "auroc": compute_auroc(y_test, scores),
        "recall_at_1pct_fpr": compute_recall_at_fpr(y_test, scores),
        "wall_clock_s": time.time() - t0,
        "error": False,
    }

    # OOD evaluation (probes trained on Enron, tested on SMS)
    if ood_activations is not None and ood_labels is not None:
        ood_scores = probe.score(ood_activations)
        result["ood_auroc"] = compute_auroc(ood_labels, ood_scores)
        result["ood_recall_at_1pct_fpr"] = compute_recall_at_fpr(ood_labels, ood_scores)
    else:
        result["ood_auroc"] = float("nan")
        result["ood_recall_at_1pct_fpr"] = float("nan")

    return result


def run_sweep(config: dict, model_filter: str = None, dataset_filter: str = None,
              probe_filter: list[str] = None):
    """Run the full experiment sweep."""
    cache_dir = config["extraction"]["cache_dir"]
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    k_values = config["sweep"]["k_values"]
    n_seeds = config["sweep"]["seeds"]
    balance_modes = config["sweep"]["balance_modes"]

    # Filter probes
    probes_to_run = {}
    for name, cls in PROBE_REGISTRY.items():
        if probe_filter and name not in probe_filter:
            continue
        probes_to_run[name] = cls

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        if model_filter and model_name != model_filter:
            continue

        layers = model_cfg["layers"]

        for ds_name, ds_hf_path in config["datasets"].items():
            if dataset_filter and ds_name != dataset_filter:
                continue

            # Skip OOD dataset as training source (SMS is OOD test only)
            if ds_name == "sms":
                continue

            print(f"\n{'='*60}")
            print(f"Model: {model_name} | Dataset: {ds_name}")
            print(f"{'='*60}")

            # Load splits
            splits_path = (
                Path(cache_dir)
                / _sanitize_model_name(model_name)
                / ds_name
                / "splits.json"
            )
            if not splits_path.exists():
                print(f"  Splits not found at {splits_path}, skipping.")
                continue
            splits = load_splits(splits_path)

            # Try to load OOD data (SMS)
            ood_acts_by_layer = {}
            ood_labels = None
            ood_ds = "sms"
            for layer in layers:
                try:
                    ood_a, ood_l = load_cached_activations(
                        cache_dir, model_name, ood_ds, layer
                    )
                    ood_acts_by_layer[layer] = ood_a
                    ood_labels = ood_l
                except FileNotFoundError:
                    pass

            # Output file for this model+dataset
            out_file = results_dir / f"{_sanitize_model_name(model_name)}_{ds_name}.jsonl"

            for layer in layers:
                for probe_name, probe_cls in probes_to_run.items():
                    # Skip SAE probe for non-Gemma models
                    if probe_name == "P5_sae" and "gemma" not in model_name.lower():
                        continue

                    # Load correct activation cache
                    suffix = _get_dataset_suffix(probe_cls)
                    cache_ds_name = f"{ds_name}{suffix}"

                    try:
                        activations, labels = load_cached_activations(
                            cache_dir, model_name, cache_ds_name, layer
                        )
                    except FileNotFoundError:
                        print(f"  Cache not found for {cache_ds_name} layer {layer}, skipping {probe_name}")
                        continue

                    # Also load the matching OOD cache
                    ood_a = None
                    ood_l = None
                    if suffix:
                        try:
                            ood_a, ood_l = load_cached_activations(
                                cache_dir, model_name, f"{ood_ds}{suffix}", layer
                            )
                        except FileNotFoundError:
                            pass
                    else:
                        ood_a = ood_acts_by_layer.get(layer)
                        ood_l = ood_labels

                    desc = f"  {probe_name} | layer {layer}"
                    total = len(k_values) * len(balance_modes) * n_seeds

                    for k in k_values:
                        for balance_mode in balance_modes:
                            for seed in range(n_seeds):
                                result = run_single_experiment(
                                    probe_cls=probe_cls,
                                    activations=activations,
                                    labels=labels,
                                    splits=splits,
                                    k=k,
                                    seed=seed,
                                    balance_mode=balance_mode,
                                    ood_activations=ood_a,
                                    ood_labels=ood_l,
                                )

                                if result is None:
                                    continue

                                result.update({
                                    "probe": probe_name,
                                    "k": k,
                                    "seed": seed,
                                    "balance_mode": balance_mode,
                                    "model": model_name,
                                    "layer": layer,
                                    "dataset": ds_name,
                                })

                                with open(out_file, "a") as f:
                                    f.write(json.dumps(result) + "\n")

                    print(f"{desc}: {total} experiments done")

    print(f"\nAll results written to {results_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run probe benchmark sweep")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--model", default=None, help="Filter to one model")
    parser.add_argument("--dataset", default=None, help="Filter to one dataset")
    parser.add_argument("--probes", nargs="+", default=None, help="Filter to specific probes")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_sweep(config, model_filter=args.model, dataset_filter=args.dataset,
              probe_filter=args.probes)


if __name__ == "__main__":
    main()
