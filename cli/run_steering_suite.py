from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from cli.run_task_sweep import _bundle_suffix
from data.task_feature_loading import load_feature_bundle
from evaluation.metrics import compute_auroc, compute_recall_at_fpr
from interventions.direction_builders import probe_direction
from interventions.online_steering import steering_hooks
from interventions.steering import SteeringConfig, steer_activation
from task_benchmark import TASK_PROBE_REGISTRY
from task_benchmark.sampling import sample_few_shot_train
from tasks import TASK_REGISTRY


def _parse_alpha_values(raw, fallback: list[float]) -> list[float]:
    if raw is None:
        return fallback
    if isinstance(raw, list):
        return [float(x) for x in raw]
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _merged_task_data(cfg: dict) -> dict[str, str]:
    if cfg.get("task_data"):
        return dict(cfg["task_data"])
    repo_cfg = Path("config.yaml")
    if repo_cfg.exists():
        root = yaml.safe_load(repo_cfg.read_text()) or {}
        return dict(root.get("task_data", {}))
    return {}


def _model_feature_map(cfg: dict) -> dict[str, dict]:
    return {model_cfg["name"]: model_cfg["feature_dirs"] for model_cfg in cfg.get("models", [])}


def _make_probe(probe_name: str, probe_cls, cfg: dict):
    if probe_name == "P5_sae":
        sae_cfg = cfg.get("sae", {})
        return probe_cls(
            sae_release=sae_cfg.get("release"),
            sae_id=sae_cfg.get("id"),
            device=sae_cfg.get("device", "cpu"),
        )
    return probe_cls()


def _quantile_threshold(scores: np.ndarray, labels: np.ndarray, quantile: float) -> float:
    neg_scores = scores[labels == 0]
    if len(neg_scores) == 0:
        return float(np.quantile(scores, quantile))
    return float(np.quantile(neg_scores, quantile))


def _direction_scores(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float) @ np.asarray(direction, dtype=float)


def _plot_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, group in df.groupby("steering_mode", dropna=False):
        ax.scatter(
            group["clean_retention_drop"],
            group["risk_suppression"],
            label=mode,
            alpha=0.8,
        )
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Clean retention drop")
    ax.set_ylabel("Risk suppression")
    ax.set_title("Steering tradeoff")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_reference_maps(examples) -> dict[str, dict[str, list[str]]]:
    refs: dict[str, dict[str, list[str]]] = {}
    for ex in examples:
        refs.setdefault(ex.question_id or ex.example_id, {"clean": [], "risky": []})
        target = "risky" if ex.label == 1 else "clean"
        if ex.final_answer:
            refs[ex.question_id or ex.example_id][target].append(_normalize_text(ex.final_answer))
        if ex.assistant_response:
            refs[ex.question_id or ex.example_id][target].append(_normalize_text(ex.assistant_response))
    return refs


def _match_any(text: str, refs: list[str]) -> bool:
    norm = _normalize_text(text)
    if not norm:
        return False
    for ref in refs:
        if not ref:
            continue
        if ref in norm or norm in ref:
            return True
    return False


def _build_prompt(example) -> str:
    parts = []
    if example.context:
        parts.append(example.context)
    parts.append(example.prompt)
    if example.task_family == "cot_distortion":
        parts.append("Think step by step, then give the final answer.")
    else:
        parts.append("Respond briefly and directly.")
    return "\n\n".join(parts)


def _load_model_and_tokenizer(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _generate_response(model, tokenizer, prompt: str, layer: int, direction: np.ndarray, alpha: float, threshold: float | None, max_new_tokens: int) -> tuple[str, dict]:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    stats_payload = {"n_forward_calls": 0, "n_applied": 0, "mean_score": float("nan")}
    with torch.no_grad():
        if alpha == 0.0:
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            with steering_hooks(model, [layer], direction=direction, alpha=alpha, threshold=threshold) as stats:
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                stats_payload = {
                    "n_forward_calls": stats.n_forward_calls,
                    "n_applied": stats.n_applied,
                    "mean_score": stats.mean_score,
                }

    new_tokens = generated[0, encoded["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, stats_payload


def _load_online_examples(task_name: str, task_data_map: dict[str, str], example_ids: list[str]):
    task_path = task_data_map.get(task_name)
    if not task_path:
        raise FileNotFoundError(f"Missing task_data path for {task_name}")
    task = TASK_REGISTRY[task_name]()
    examples = task.load(task_path)
    by_id = {ex.example_id: ex for ex in examples}
    selected = [by_id[example_id] for example_id in example_ids if example_id in by_id]
    return examples, selected


def _run_feature_space(cfg: dict, results_dir: Path, best: pd.DataFrame, alpha_values: list[float], steering_seed: int, threshold_quantile: float) -> pd.DataFrame:
    feature_map = _model_feature_map(cfg)
    rows = []
    for row in best.itertuples(index=False):
        probe_name = row.probe
        probe_cls = TASK_PROBE_REGISTRY.get(probe_name)
        if probe_cls is None:
            continue
        model_features = feature_map.get(row.model)
        if not model_features:
            continue
        source_dir = model_features.get(row.source_task)
        target_dir = model_features.get(row.target_task)
        if not source_dir or not target_dir:
            continue

        suffix = _bundle_suffix(probe_cls)
        try:
            source_train = load_feature_bundle(source_dir, "train", int(row.layer), cache_suffix=suffix)
            source_eval = load_feature_bundle(source_dir, "eval", int(row.layer), cache_suffix=suffix)
            target_test = load_feature_bundle(target_dir, "test", int(row.layer), cache_suffix=suffix)
        except FileNotFoundError:
            continue
        if row.view not in source_train or row.view not in source_eval or row.view not in target_test:
            continue

        X_train = source_train[row.view]
        y_train = source_train["labels"]
        try:
            X_fs, y_fs = sample_few_shot_train(X_train, y_train, k=int(row.k), seed=steering_seed, balance_mode=row.balance_mode)
        except ValueError:
            continue

        probe = _make_probe(probe_name, probe_cls, cfg)
        try:
            probe.fit(X_fs, y_fs)
            direction = probe_direction(probe, X_fs, y_fs)
        except Exception:
            continue

        eval_scores = _direction_scores(source_eval[row.view], direction)
        threshold = _quantile_threshold(eval_scores, source_eval["labels"], threshold_quantile)
        X_target = target_test[row.view]
        y_target = target_test["labels"]
        base_scores = _direction_scores(X_target, direction)
        base_recall = float(np.mean(base_scores[y_target == 1] > threshold)) if np.any(y_target == 1) else float("nan")
        base_fpr = float(np.mean(base_scores[y_target == 0] > threshold)) if np.any(y_target == 0) else float("nan")
        base_pos_mean = float(np.mean(base_scores[y_target == 1])) if np.any(y_target == 1) else float("nan")
        base_neg_mean = float(np.mean(base_scores[y_target == 0])) if np.any(y_target == 0) else float("nan")

        for steering_mode in ["unconditional", "threshold_triggered"]:
            mode_threshold = None if steering_mode == "unconditional" else threshold
            for alpha in alpha_values:
                cfg_obj = SteeringConfig(alpha=float(alpha), threshold=mode_threshold)
                steered_X = np.stack(
                    [steer_activation(vec, direction, float(score), cfg_obj) for vec, score in zip(X_target, base_scores)],
                    axis=0,
                )
                steered_scores = _direction_scores(steered_X, direction)
                steered_recall = float(np.mean(steered_scores[y_target == 1] > threshold)) if np.any(y_target == 1) else float("nan")
                steered_fpr = float(np.mean(steered_scores[y_target == 0] > threshold)) if np.any(y_target == 0) else float("nan")
                steered_pos_mean = float(np.mean(steered_scores[y_target == 1])) if np.any(y_target == 1) else float("nan")
                steered_neg_mean = float(np.mean(steered_scores[y_target == 0])) if np.any(y_target == 0) else float("nan")
                rows.append(
                    {
                        "steering_backend": "feature_space",
                        "model": row.model,
                        "source_task": row.source_task,
                        "target_task": row.target_task,
                        "probe": probe_name,
                        "layer": int(row.layer),
                        "view": row.view,
                        "k": int(row.k),
                        "balance_mode": row.balance_mode,
                        "steering_mode": steering_mode,
                        "alpha": float(alpha),
                        "threshold": threshold,
                        "risk_suppression": base_recall - steered_recall,
                        "clean_retention_drop": steered_fpr - base_fpr,
                        "suppression_at_threshold": base_recall - steered_recall,
                        "fpr_change": steered_fpr - base_fpr,
                        "positive_score_shift": steered_pos_mean - base_pos_mean,
                        "negative_score_shift": steered_neg_mean - base_neg_mean,
                        "selectivity_score": (base_recall - steered_recall) - abs(steered_fpr - base_fpr),
                    }
                )
    return pd.DataFrame(rows)


def _run_online(cfg: dict, results_dir: Path, best: pd.DataFrame, alpha_values: list[float], steering_seed: int, threshold_quantile: float, max_new_tokens: int) -> pd.DataFrame:
    feature_map = _model_feature_map(cfg)
    task_data_map = _merged_task_data(cfg)
    model_cache = {}
    rows = []

    for row in best.itertuples(index=False):
        probe_name = row.probe
        probe_cls = TASK_PROBE_REGISTRY.get(probe_name)
        if probe_cls is None:
            continue
        model_features = feature_map.get(row.model)
        if not model_features:
            continue
        source_dir = model_features.get(row.source_task)
        target_dir = model_features.get(row.target_task)
        if not source_dir or not target_dir:
            continue

        suffix = _bundle_suffix(probe_cls)
        try:
            source_train = load_feature_bundle(source_dir, "train", int(row.layer), cache_suffix=suffix)
            source_eval = load_feature_bundle(source_dir, "eval", int(row.layer), cache_suffix=suffix)
            target_test = load_feature_bundle(target_dir, "test", int(row.layer), cache_suffix=suffix)
        except FileNotFoundError:
            continue
        if row.view not in source_train or row.view not in source_eval:
            continue

        X_train = source_train[row.view]
        y_train = source_train["labels"]
        try:
            X_fs, y_fs = sample_few_shot_train(X_train, y_train, k=int(row.k), seed=steering_seed, balance_mode=row.balance_mode)
        except ValueError:
            continue

        probe = _make_probe(probe_name, probe_cls, cfg)
        try:
            probe.fit(X_fs, y_fs)
            direction = probe_direction(probe, X_fs, y_fs)
        except Exception:
            continue
        if not np.any(direction):
            continue

        eval_scores = _direction_scores(source_eval[row.view], direction)
        threshold = _quantile_threshold(eval_scores, source_eval["labels"], threshold_quantile)

        try:
            all_target_examples, selected_examples = _load_online_examples(
                row.target_task,
                task_data_map,
                list(target_test["example_ids"].astype(str)),
            )
        except Exception:
            continue
        if not selected_examples:
            continue
        ref_map = _build_reference_maps(all_target_examples)

        if row.model not in model_cache:
            try:
                model_cache[row.model] = _load_model_and_tokenizer(row.model)
            except Exception:
                continue
        model, tokenizer = model_cache[row.model]

        baseline_cache = {}
        for ex in selected_examples:
            prompt = _build_prompt(ex)
            response, stats = _generate_response(
                model,
                tokenizer,
                prompt,
                layer=int(row.layer),
                direction=direction,
                alpha=0.0,
                threshold=None,
                max_new_tokens=max_new_tokens,
            )
            baseline_cache[ex.example_id] = (response, stats)

        for steering_mode in ["unconditional", "threshold_triggered"]:
            mode_threshold = None if steering_mode == "unconditional" else float(threshold)
            for alpha in alpha_values:
                risky_base = []
                risky_steered = []
                clean_base = []
                clean_steered = []
                apply_rates = []
                live_scores = []
                for ex in selected_examples:
                    prompt = _build_prompt(ex)
                    base_text, _ = baseline_cache[ex.example_id]
                    steered_text, stats = _generate_response(
                        model,
                        tokenizer,
                        prompt,
                        layer=int(row.layer),
                        direction=direction,
                        alpha=float(alpha),
                        threshold=mode_threshold,
                        max_new_tokens=max_new_tokens,
                    )
                    refs = ref_map.get(ex.question_id or ex.example_id, {"clean": [], "risky": []})
                    base_risky = float(_match_any(base_text, refs["risky"]))
                    steered_risky = float(_match_any(steered_text, refs["risky"]))
                    base_clean = float(_match_any(base_text, refs["clean"]))
                    steered_clean = float(_match_any(steered_text, refs["clean"]))
                    if ex.label == 1:
                        risky_base.append(base_risky)
                        risky_steered.append(steered_risky)
                    else:
                        clean_base.append(base_clean)
                        clean_steered.append(steered_clean)
                    if stats["n_forward_calls"] > 0:
                        apply_rates.append(stats["n_applied"] / stats["n_forward_calls"])
                        live_scores.append(stats["mean_score"])

                base_risky_rate = float(np.mean(risky_base)) if risky_base else float("nan")
                steered_risky_rate = float(np.mean(risky_steered)) if risky_steered else float("nan")
                base_clean_rate = float(np.mean(clean_base)) if clean_base else float("nan")
                steered_clean_rate = float(np.mean(clean_steered)) if clean_steered else float("nan")
                risk_suppression = base_risky_rate - steered_risky_rate if not np.isnan(base_risky_rate) and not np.isnan(steered_risky_rate) else float("nan")
                clean_retention_drop = base_clean_rate - steered_clean_rate if not np.isnan(base_clean_rate) and not np.isnan(steered_clean_rate) else float("nan")
                rows.append(
                    {
                        "steering_backend": "online_hook",
                        "model": row.model,
                        "source_task": row.source_task,
                        "target_task": row.target_task,
                        "probe": probe_name,
                        "layer": int(row.layer),
                        "view": row.view,
                        "k": int(row.k),
                        "balance_mode": row.balance_mode,
                        "steering_mode": steering_mode,
                        "alpha": float(alpha),
                        "threshold": threshold,
                        "base_risky_rate": base_risky_rate,
                        "steered_risky_rate": steered_risky_rate,
                        "base_clean_rate": base_clean_rate,
                        "steered_clean_rate": steered_clean_rate,
                        "risk_suppression": risk_suppression,
                        "clean_retention_drop": clean_retention_drop,
                        "suppression_at_threshold": risk_suppression,
                        "fpr_change": clean_retention_drop,
                        "mean_hook_apply_rate": float(np.mean(apply_rates)) if apply_rates else float("nan"),
                        "mean_live_score": float(np.mean(live_scores)) if live_scores else float("nan"),
                        "n_positive_examples": len(risky_base),
                        "n_negative_examples": len(clean_base),
                        "selectivity_score": risk_suppression - max(clean_retention_drop, 0.0) if not np.isnan(risk_suppression) and not np.isnan(clean_retention_drop) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run steering experiments from best task systems")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--selection_file", default="task_best_view_layer.csv")
    parser.add_argument("--steering_seed", type=int, default=0)
    parser.add_argument("--threshold_quantile", type=float, default=0.99)
    parser.add_argument("--alpha_values", default=None)
    parser.add_argument("--backend", default=None, choices=["online", "feature_space"])
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    results_dir = Path(args.results_dir or cfg["results_dir"])
    best_path = results_dir / args.selection_file
    if not best_path.exists() or best_path.stat().st_size == 0:
        print("No best-system file found for steering.")
        return
    best = pd.read_csv(best_path)
    if best.empty:
        print("Empty best-system file; skipping steering.")
        return

    alpha_values = _parse_alpha_values(args.alpha_values, cfg.get("intervention", {}).get("alpha_values", [0.25, 0.5, 1.0, 2.0]))
    backend = args.backend or cfg.get("intervention", {}).get("backend", "online")
    if backend == "online" and best["model"].astype(str).str.startswith("dummy-").any():
        backend = "feature_space"

    if backend == "online":
        out = _run_online(cfg, results_dir, best, alpha_values, args.steering_seed, args.threshold_quantile, args.max_new_tokens)
    else:
        out = _run_feature_space(cfg, results_dir, best, alpha_values, args.steering_seed, args.threshold_quantile)

    out_path = results_dir / "task_steering_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"saved {out_path}")

    if not out.empty:
        idx = out.groupby(["model", "source_task", "target_task", "steering_mode"], dropna=False)["selectivity_score"].idxmax()
        best_out = out.loc[idx].reset_index(drop=True)
    else:
        best_out = pd.DataFrame()
    best_path = results_dir / "task_steering_best.csv"
    best_out.to_csv(best_path, index=False)
    print(f"saved {best_path}")

    plot_path = results_dir / "task_steering_tradeoff.png"
    _plot_tradeoff(out, plot_path)
    if plot_path.exists():
        print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
