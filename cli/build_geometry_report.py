from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cli.common import load_yaml
from data.task_feature_loading import infer_layers, load_feature_bundle
from evaluation.geometry import build_direction_alignment, compute_geometry_metrics, mean_difference_direction


def _iter_feature_dirs(cfg: dict) -> list[tuple[str, str, str]]:
    rows = []
    for model_cfg in cfg.get("models", []):
        model = model_cfg["name"]
        for task, feature_dir in model_cfg.get("feature_dirs", {}).items():
            rows.append((model, task, str(feature_dir)))
    return rows


def _activation_views(bundle: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    """Return only dense activation matrices, never per-example metadata."""

    n_examples = len(bundle["labels"])
    views: list[tuple[str, np.ndarray]] = []
    for name, values in bundle.items():
        array = np.asarray(values)
        if (
            array.ndim == 2
            and len(array) == n_examples
            and np.issubdtype(array.dtype, np.number)
        ):
            views.append((name, array))
    return views


def main() -> None:
    parser = argparse.ArgumentParser(description="Build geometry and direction-alignment reports from structured task features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    results_dir = Path(args.results_dir or cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    geometry_rows = []
    direction_rows = []
    for model, task, feature_dir in _iter_feature_dirs(cfg):
        layers = infer_layers(feature_dir)
        for layer in layers:
            bundle = load_feature_bundle(feature_dir, args.split, layer)
            y = bundle["labels"]
            activation_views = _activation_views(bundle)
            if not activation_views:
                raise ValueError(
                    f"No numeric activation views found in "
                    f"{feature_dir}/{args.split}_layer{layer}.npz"
                )
            for view, X in activation_views:
                metrics = compute_geometry_metrics(X, y)
                geometry_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "split": args.split,
                        "layer": layer,
                        "view": view,
                        **metrics,
                    }
                )
                direction_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "split": args.split,
                        "layer": layer,
                        "view": view,
                        "direction": mean_difference_direction(X, y),
                    }
                )

    geometry = pd.DataFrame(geometry_rows)
    geometry_path = results_dir / "task_geometry_summary.csv"
    geometry.to_csv(geometry_path, index=False)
    print(f"saved {geometry_path}")

    alignment = build_direction_alignment(direction_rows)
    alignment_path = results_dir / "task_direction_alignment.csv"
    alignment.to_csv(alignment_path, index=False)
    print(f"saved {alignment_path}")

    best_path = results_dir / "task_best_view_layer.csv"
    if best_path.exists() and best_path.stat().st_size > 0:
        best = pd.read_csv(best_path)
        best = best.copy()
        best["task_pair_min"] = best[["source_task", "target_task"]].min(axis=1)
        best["task_pair_max"] = best[["source_task", "target_task"]].max(axis=1)
        source_geo = geometry.rename(columns={c: f"source_{c}" for c in geometry.columns if c not in {"model", "layer", "view"}})
        source_geo = source_geo.rename(columns={"source_layer": "layer", "source_view": "view", "source_model": "model"})
        target_geo = geometry.rename(columns={c: f"target_{c}" for c in geometry.columns if c not in {"model", "layer", "view"}})
        target_geo = target_geo.rename(columns={"target_layer": "layer", "target_view": "view", "target_model": "model"})

        joined = best.merge(source_geo, how="left", left_on=["model", "source_task", "layer", "view"], right_on=["model", "source_task", "layer", "view"])
        joined = joined.merge(target_geo, how="left", left_on=["model", "target_task", "layer", "view"], right_on=["model", "target_task", "layer", "view"])
        if not alignment.empty:
            joined = joined.merge(
                alignment,
                how="left",
                left_on=["model", "layer", "view", "task_pair_min", "task_pair_max"],
                right_on=["model", "layer", "view", "task_pair_min", "task_pair_max"],
            )
        joined_path = results_dir / "task_geometry_benchmark_join.csv"
        joined.to_csv(joined_path, index=False)
        print(f"saved {joined_path}")

        corr_rows = []
        outcome_cols = [
            c
            for c in [
                "transfer_tpr_at_1pct_reference_alert_budget_mean",
                "transfer_auroc_mean",
                "test_tpr_at_1pct_reference_alert_budget_mean",
            ]
            if c in joined.columns
        ]
        metric_cols = [
            c for c in joined.columns
            if c.startswith("source_") or c.startswith("target_")
        ]
        for metric in metric_cols:
            if metric.endswith(("_task", "_split")):
                continue
            if not pd.api.types.is_numeric_dtype(joined[metric]):
                continue
            for outcome in outcome_cols:
                valid = joined[[metric, outcome]].dropna()
                if len(valid) < 2:
                    continue
                if float(valid[metric].std(ddof=0)) == 0.0 or float(valid[outcome].std(ddof=0)) == 0.0:
                    continue
                corr_rows.append(
                    {
                        "geometry_metric": metric,
                        "outcome_metric": outcome,
                        "pearson_correlation": float(valid[metric].corr(valid[outcome])),
                        "n_rows": int(len(valid)),
                    }
                )
        corr_path = results_dir / "task_geometry_correlations.csv"
        pd.DataFrame(corr_rows).to_csv(corr_path, index=False)
        print(f"saved {corr_path}")


if __name__ == "__main__":
    main()
