"""Sweep orchestrator for probe benchmark experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from data.splitting import get_split_arrays, load_splits, sample_train_set
from extraction.extractor import _sanitize_model_name, load_cached_activations
from evaluation.metrics import compute_auroc, compute_recall_at_fpr
from probes import PROBE_REGISTRY


def _get_dataset_suffix(probe_cls) -> str:
    mod = probe_cls.requires_modified_activations
    if mod is None:
        return ""
    return f"_{mod}"


def _load_existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            run_id = row.get("run_id")
            if run_id:
                seen.add(run_id)
    return seen


def _save_prediction_artifact(
    predictions_dir: Path,
    run_id: str,
    payload: dict[str, np.ndarray | None],
) -> None:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    safe_payload = {}
    for k, v in payload.items():
        if v is None:
            continue
        safe_payload[k] = v
    np.savez_compressed(predictions_dir / f"{run_id}.npz", **safe_payload)


def _sanitize_nans(arr: np.ndarray, name: str = "activations") -> np.ndarray:
    """Replace NaN rows with zeros once, returning the (possibly copied) array."""
    nan_mask = np.isnan(arr).any(axis=1)
    if nan_mask.any():
        import logging
        logging.warning(
            "Found %d / %d samples with NaN %s — replacing with zeros",
            int(nan_mask.sum()), len(arr), name,
        )
        arr = arr.copy()
        arr[nan_mask] = 0.0
    return arr


def run_single_experiment(
    probe_cls,
    activations: np.ndarray,
    labels: np.ndarray,
    splits: dict,
    k: int,
    seed: int,
    balance_mode: str,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ood_activations: np.ndarray | None = None,
    ood_labels: np.ndarray | None = None,
) -> tuple[dict, dict] | tuple[None, None]:
    """Train one probe, score eval, test, and optional OOD."""
    t0 = time.time()

    try:
        X_train, y_train = sample_train_set(
            activations=activations,
            labels=labels,
            train_indices=splits["train_pool"],
            k=k,
            seed=seed,
            balance_mode=balance_mode,
        )
    except ValueError:
        return None, None

    probe = probe_cls()

    try:
        probe.fit(X_train, y_train)
    except Exception as err:
        result = {
            "eval_auroc": float("nan"),
            "eval_recall_at_1pct_fpr": float("nan"),
            "test_auroc": float("nan"),
            "test_recall_at_1pct_fpr": float("nan"),
            "ood_auroc": float("nan"),
            "ood_recall_at_1pct_fpr": float("nan"),
            "n_train_pos": int((y_train == 1).sum()),
            "n_train_neg": int((y_train == 0).sum()),
            "wall_clock_s": time.time() - t0,
            "error": True,
            "error_type": type(err).__name__,
        }
        return result, {}

    eval_scores = probe.score(X_eval)
    test_scores = probe.score(X_test)

    result = {
        "eval_auroc": compute_auroc(y_eval, eval_scores),
        "eval_recall_at_1pct_fpr": compute_recall_at_fpr(y_eval, eval_scores),
        "test_auroc": compute_auroc(y_test, test_scores),
        "test_recall_at_1pct_fpr": compute_recall_at_fpr(y_test, test_scores),
        "n_train_pos": int((y_train == 1).sum()),
        "n_train_neg": int((y_train == 0).sum()),
        "wall_clock_s": time.time() - t0,
        "error": False,
        "error_type": None,
    }

    pred_payload = {
        "y_eval": y_eval,
        "eval_scores": eval_scores,
        "y_test": y_test,
        "test_scores": test_scores,
    }

    if ood_activations is not None and ood_labels is not None:
        ood_scores = probe.score(ood_activations)
        result["ood_auroc"] = compute_auroc(ood_labels, ood_scores)
        result["ood_recall_at_1pct_fpr"] = compute_recall_at_fpr(ood_labels, ood_scores)
        pred_payload["y_ood"] = ood_labels
        pred_payload["ood_scores"] = ood_scores
    else:
        result["ood_auroc"] = float("nan")
        result["ood_recall_at_1pct_fpr"] = float("nan")

    return result, pred_payload


def run_sweep(
    config: dict,
    model_filter: str | None = None,
    dataset_filter: str | None = None,
    probe_filter: list[str] | None = None,
) -> None:
    cache_dir = config["extraction"]["cache_dir"]
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions_dir = results_dir / "predictions"
    save_predictions = config.get("results", {}).get("save_predictions", True)
    overwrite_results = config.get("results", {}).get("overwrite", False)

    k_values = config["sweep"]["k_values"]
    n_seeds = config["sweep"]["seeds"]
    balance_modes = config["sweep"]["balance_modes"]

    probes_to_run = {
        name: cls
        for name, cls in PROBE_REGISTRY.items()
        if not probe_filter or name in probe_filter
    }

    dataset_cfg = config.get("datasets", {})
    if isinstance(dataset_cfg, dict) and "in_distribution" in dataset_cfg:
        train_datasets = dataset_cfg["in_distribution"]
        ood_map = dataset_cfg.get("ood", {})
    else:
        train_datasets = [k for k in dataset_cfg.keys() if k != "sms"]
        ood_map = {"enron": "sms"}

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        if model_filter and model_name != model_filter:
            continue

        layers = model_cfg["layers"]

        for ds_name in train_datasets:
            if dataset_filter and ds_name != dataset_filter:
                continue

            print(f"\n{'=' * 72}")
            print(f"Model: {model_name} | Dataset: {ds_name}")
            print(f"{'=' * 72}")

            splits_path = (
                Path(cache_dir)
                / _sanitize_model_name(model_name)
                / ds_name
                / "splits.json"
            )
            if not splits_path.exists():
                print(f"  Missing splits at {splits_path}. Skipping.")
                continue
            splits = load_splits(splits_path)

            out_file = results_dir / f"{_sanitize_model_name(model_name)}_{ds_name}.jsonl"
            existing_run_ids = set() if overwrite_results else _load_existing_run_ids(out_file)
            if overwrite_results and out_file.exists():
                out_file.unlink()

            ood_ds = ood_map.get(ds_name)

            # Cache loaded & NaN-sanitised activations by (dataset_name, layer)
            _act_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

            def _load_and_sanitize(
                ds: str, layer: int, _cache: dict = _act_cache
            ) -> tuple[np.ndarray, np.ndarray]:
                key = (ds, layer)
                if key not in _cache:
                    acts, lbls = load_cached_activations(cache_dir, model_name, ds, layer)
                    acts = _sanitize_nans(acts, name=f"{ds}/layer_{layer}")
                    _cache[key] = (acts, lbls)
                return _cache[key]

            # Pre-load base OOD activations
            if ood_ds:
                for layer in layers:
                    try:
                        _load_and_sanitize(ood_ds, layer)
                    except FileNotFoundError:
                        pass

            for layer in layers:
                for probe_name, probe_cls in probes_to_run.items():
                    if probe_name == "P5_sae" and "gemma" not in model_name.lower():
                        continue

                    suffix = _get_dataset_suffix(probe_cls)
                    cache_ds_name = f"{ds_name}{suffix}"

                    try:
                        activations, labels = _load_and_sanitize(cache_ds_name, layer)
                    except FileNotFoundError:
                        print(f"  Missing cache for {cache_ds_name} layer {layer}. Skipping {probe_name}.")
                        continue

                    # Precompute eval/test splits once per probe+layer
                    X_eval, y_eval = get_split_arrays(activations, labels, splits["eval"])
                    X_test, y_test = get_split_arrays(activations, labels, splits["test"])

                    ood_a = None
                    ood_l = None
                    if ood_ds:
                        ood_cache_name = f"{ood_ds}{suffix}" if suffix else ood_ds
                        try:
                            ood_a, ood_l = _load_and_sanitize(ood_cache_name, layer)
                        except FileNotFoundError:
                            pass

                    n_done = 0
                    total = len(k_values) * len(balance_modes) * n_seeds
                    row_buffer: list[str] = []

                    for k in k_values:
                        for balance_mode in balance_modes:
                            for seed in range(n_seeds):
                                run_id = (
                                    f"{_sanitize_model_name(model_name)}"
                                    f"__{ds_name}"
                                    f"__{probe_name}"
                                    f"__layer{layer}"
                                    f"__k{k}"
                                    f"__seed{seed}"
                                    f"__{balance_mode}"
                                )
                                if run_id in existing_run_ids:
                                    continue

                                result, pred_payload = run_single_experiment(
                                    probe_cls=probe_cls,
                                    activations=activations,
                                    labels=labels,
                                    splits=splits,
                                    k=k,
                                    seed=seed,
                                    balance_mode=balance_mode,
                                    X_eval=X_eval,
                                    y_eval=y_eval,
                                    X_test=X_test,
                                    y_test=y_test,
                                    ood_activations=ood_a,
                                    ood_labels=ood_l,
                                )
                                if result is None:
                                    continue

                                row = {
                                    "run_id": run_id,
                                    "probe": probe_name,
                                    "k": int(k),
                                    "seed": int(seed),
                                    "balance_mode": balance_mode,
                                    "model": model_name,
                                    "layer": int(layer),
                                    "dataset": ds_name,
                                }
                                row.update(result)
                                row_buffer.append(json.dumps(row))

                                if save_predictions and pred_payload:
                                    _save_prediction_artifact(predictions_dir, run_id, pred_payload)

                                existing_run_ids.add(run_id)
                                n_done += 1

                    # Flush buffered rows once per probe+layer
                    if row_buffer:
                        with open(out_file, "a", encoding="utf-8") as f:
                            f.write("\n".join(row_buffer) + "\n")

                    print(f"  {probe_name} | layer {layer}: completed {n_done} new runs out of {total}")

            # Free cached activations for this model+dataset
            _act_cache.clear()

    print(f"\nResults saved under {results_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probe benchmark sweep")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--probes", nargs="+", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_sweep(
        config=config,
        model_filter=args.model,
        dataset_filter=args.dataset,
        probe_filter=args.probes,
    )


if __name__ == "__main__":
    main()
