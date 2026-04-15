from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from data.task_feature_loading import infer_layers, load_feature_bundle
from evaluation.metrics import compute_auprc, compute_auroc, compute_brier_score, compute_ece, compute_recall_at_fpr
from task_benchmark import TASK_PROBE_REGISTRY
from task_benchmark.sampling import sample_few_shot_train


def _load_existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if "run_id" in row:
                    seen.add(row["run_id"])
    return seen


def _score_split(probe, bundle: Dict[str, np.ndarray], view: str) -> tuple[np.ndarray, np.ndarray]:
    X = bundle[view]
    y = bundle["labels"]
    scores = probe.score(X)
    return y, scores


def _bundle_suffix(probe_cls) -> str:
    mod = getattr(probe_cls, "requires_modified_activations", None)
    if mod is None:
        return ""
    return f"_{mod}"


def _make_probe(probe_cls, sae_release: str | None = None, sae_id: str | None = None, sae_device: str = "cpu"):
    if getattr(probe_cls, "name", "") == "P5_sae":
        return probe_cls(sae_release=sae_release, sae_id=sae_id, device=sae_device)
    return probe_cls()


def _run_one(
    probe_cls,
    source_train: Dict[str, np.ndarray],
    source_eval: Dict[str, np.ndarray],
    source_test: Dict[str, np.ndarray],
    target_test: Optional[Dict[str, np.ndarray]],
    view: str,
    k: int,
    seed: int,
    balance_mode: str,
    probe_kwargs: Optional[dict] = None,
) -> dict | None:
    t0 = time.time()
    X_train = source_train[view]
    y_train = source_train["labels"]

    try:
        X_fs, y_fs = sample_few_shot_train(X_train, y_train, k=k, seed=seed, balance_mode=balance_mode)
    except ValueError:
        return None

    probe = probe_cls(**(probe_kwargs or {}))
    try:
        probe.fit(X_fs, y_fs)
    except Exception as err:
        return {
            "eval_auroc": float("nan"),
            "eval_auprc": float("nan"),
            "eval_recall_at_1pct_fpr": float("nan"),
            "eval_brier": float("nan"),
            "eval_ece": float("nan"),
            "test_auroc": float("nan"),
            "test_auprc": float("nan"),
            "test_recall_at_1pct_fpr": float("nan"),
            "test_brier": float("nan"),
            "test_ece": float("nan"),
            "transfer_auroc": float("nan"),
            "transfer_auprc": float("nan"),
            "transfer_recall_at_1pct_fpr": float("nan"),
            "transfer_brier": float("nan"),
            "transfer_ece": float("nan"),
            "n_train_pos": int((y_fs == 1).sum()),
            "n_train_neg": int((y_fs == 0).sum()),
            "wall_clock_s": time.time() - t0,
            "error": True,
            "error_type": type(err).__name__,
        }

    y_eval, eval_scores = _score_split(probe, source_eval, view)
    y_test, test_scores = _score_split(probe, source_test, view)

    row = {
        "eval_auroc": compute_auroc(y_eval, eval_scores),
        "eval_auprc": compute_auprc(y_eval, eval_scores),
        "eval_recall_at_1pct_fpr": compute_recall_at_fpr(y_eval, eval_scores),
        "eval_brier": compute_brier_score(y_eval, eval_scores),
        "eval_ece": compute_ece(y_eval, eval_scores),
        "test_auroc": compute_auroc(y_test, test_scores),
        "test_auprc": compute_auprc(y_test, test_scores),
        "test_recall_at_1pct_fpr": compute_recall_at_fpr(y_test, test_scores),
        "test_brier": compute_brier_score(y_test, test_scores),
        "test_ece": compute_ece(y_test, test_scores),
        "n_train_pos": int((y_fs == 1).sum()),
        "n_train_neg": int((y_fs == 0).sum()),
        "wall_clock_s": time.time() - t0,
        "error": False,
        "error_type": None,
    }

    if target_test is not None and view in target_test:
        y_transfer, transfer_scores = _score_split(probe, target_test, view)
        row["transfer_auroc"] = compute_auroc(y_transfer, transfer_scores)
        row["transfer_auprc"] = compute_auprc(y_transfer, transfer_scores)
        row["transfer_recall_at_1pct_fpr"] = compute_recall_at_fpr(y_transfer, transfer_scores)
        row["transfer_brier"] = compute_brier_score(y_transfer, transfer_scores)
        row["transfer_ece"] = compute_ece(y_transfer, transfer_scores)
    else:
        row["transfer_auroc"] = float("nan")
        row["transfer_auprc"] = float("nan")
        row["transfer_recall_at_1pct_fpr"] = float("nan")
        row["transfer_brier"] = float("nan")
        row["transfer_ece"] = float("nan")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured task-benchmark sweep")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--source_task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--target_dir", default=None)
    parser.add_argument("--target_task", default=None)
    parser.add_argument("--views", default="full_text,answer")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--probes", default="P1_logistic,P2_mass_mean,P3_lda,P4_cosine,P7_mahalanobis")
    parser.add_argument("--k_values", default="1,2,4,8")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--balance_modes", default="balanced,imbalanced")
    parser.add_argument("--sae_release", default=None)
    parser.add_argument("--sae_id", default=None)
    parser.add_argument("--sae_device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    balance_modes = [x.strip() for x in args.balance_modes.split(",") if x.strip()]
    probe_names = [p.strip() for p in args.probes.split(",") if p.strip()]

    if args.layers == "all":
        layers = infer_layers(args.source_dir)
    else:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{args.source_task}__to__{args.target_task or args.source_task}.jsonl"
    existing_run_ids = set() if args.overwrite else _load_existing_run_ids(out_file)
    if args.overwrite and out_file.exists():
        out_file.unlink()

    row_buffer: List[str] = []
    total = 0
    completed = 0
    bundle_cache: Dict[tuple[str, str, int, str], Dict[str, np.ndarray]] = {}

    def _load_cached(features_dir: str, split: str, layer: int, suffix: str) -> Dict[str, np.ndarray]:
        key = (features_dir, split, layer, suffix)
        if key not in bundle_cache:
            bundle_cache[key] = load_feature_bundle(features_dir, split, layer, cache_suffix=suffix)
        return bundle_cache[key]

    for layer in layers:
        for probe_name in probe_names:
            probe_cls = TASK_PROBE_REGISTRY[probe_name]
            suffix = _bundle_suffix(probe_cls)
            try:
                source_train = _load_cached(args.source_dir, "train", layer, suffix)
                source_eval = _load_cached(args.source_dir, "eval", layer, suffix)
                source_test = _load_cached(args.source_dir, "test", layer, suffix)
                target_test = None
                if args.target_dir:
                    target_test = _load_cached(args.target_dir, "test", layer, suffix)
            except FileNotFoundError:
                print(f"missing {probe_name} cache for layer {layer}{suffix}; skipping")
                continue

            available_views = [v for v in views if v in source_train and v in source_eval and v in source_test]
            probe_kwargs = None
            if probe_name == "P5_sae":
                probe_kwargs = {
                    "sae_release": args.sae_release,
                    "sae_id": args.sae_id,
                    "device": args.sae_device,
                }
            for view in available_views:
                for k in k_values:
                    for balance_mode in balance_modes:
                        for seed in range(args.seeds):
                            total += 1
                            run_id = (
                                f"{args.model}__{args.source_task}__{args.target_task or args.source_task}"
                                f"__{probe_name}__layer{layer}__{view}__k{k}__seed{seed}__{balance_mode}"
                            )
                            if run_id in existing_run_ids:
                                continue
                            row = _run_one(
                                probe_cls=probe_cls,
                                source_train=source_train,
                                source_eval=source_eval,
                                source_test=source_test,
                                target_test=target_test,
                                view=view,
                                k=k,
                                seed=seed,
                                balance_mode=balance_mode,
                                probe_kwargs=probe_kwargs,
                            )
                            if row is None:
                                continue
                            row.update(
                                {
                                    "run_id": run_id,
                                    "probe": probe_name,
                                    "k": k,
                                    "seed": seed,
                                    "balance_mode": balance_mode,
                                    "model": args.model,
                                    "layer": layer,
                                    "view": view,
                                    "source_task": args.source_task,
                                    "target_task": args.target_task or args.source_task,
                                }
                            )
                            row_buffer.append(json.dumps(row))
                            existing_run_ids.add(run_id)
                            completed += 1

    if row_buffer:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write("\n".join(row_buffer) + "\n")

    print(f"completed {completed} runs out of {total}")
    print(f"saved results to {out_file}")


if __name__ == "__main__":
    main()
