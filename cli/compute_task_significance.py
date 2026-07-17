from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evaluation.aggregation import collect_results
from evaluation.hierarchical_statistics import (
    hierarchical_paired_rate_difference,
    holm_adjust,
)


PRIMARY_COMPARISON_ROLE = "primary_white_box_gain"
PRIMARY_IDENTITY_KEYS = ("probe", "balance_mode", "layer", "view")


def _filter_rows(df: pd.DataFrame, filters: dict[str, Any], comparison_id: str) -> pd.DataFrame:
    selected = df
    for key, value in filters.items():
        if key not in selected.columns:
            raise ValueError(f"Comparison {comparison_id} filters on absent column {key!r}")
        selected = selected[selected[key].astype(str) == str(value)]
    if selected.empty:
        raise ValueError(f"Comparison {comparison_id} selected no runs for filters {filters}")
    duplicate_seeds = selected["seed"].astype(str).duplicated(keep=False)
    if duplicate_seeds.any():
        raise ValueError(
            f"Comparison {comparison_id} does not identify exactly one run per seed: {filters}"
        )
    return selected.copy()


def _validate_primary_source_selection(
    comparison: dict[str, Any],
    filters_a: dict[str, Any],
    filters_b: dict[str, Any],
    selected: pd.DataFrame,
    comparison_id: str,
) -> None:
    if comparison.get("comparison_role") != PRIMARY_COMPARISON_ROLE:
        return
    for filters, access_regime, label in (
        (filters_a, "white_box", "system_a"),
        (filters_b, "black_box", "system_b"),
    ):
        missing = [
            key
            for key in ("model", "source_task", *PRIMARY_IDENTITY_KEYS)
            if key not in filters
        ]
        if missing:
            raise ValueError(
                f"Primary comparison {comparison_id} {label} lacks exact "
                f"source-selected identity fields {missing}"
            )
        candidate = selected[
            (selected["model"].astype(str) == str(filters["model"]))
            & (selected["source_task"].astype(str) == str(filters["source_task"]))
            & (selected["access_regime"].astype(str) == access_regime)
        ]
        if len(candidate) != 1:
            raise ValueError(
                f"Primary comparison {comparison_id} expected one {access_regime} "
                f"source selection, found {len(candidate)}"
            )
        expected = candidate.iloc[0]
        mismatched = {
            key: {"registered": filters[key], "source_selected": expected[key]}
            for key in PRIMARY_IDENTITY_KEYS
            if str(filters[key]) != str(expected[key])
        }
        if mismatched:
            raise ValueError(
                f"Primary comparison {comparison_id} {label} does not match the "
                f"source-only selector: {mismatched}"
            )


def _load_prediction_file(path_value: str, results_dir: Path) -> pd.DataFrame:
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([results_dir / path, results_dir.parent / path])
    path = next((candidate for candidate in candidates if candidate.exists()), path)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction artifact {path_value}")
    return pd.read_json(path, lines=True)


def _paired_predictions(
    runs_a: pd.DataFrame,
    runs_b: pd.DataFrame,
    split: str,
    results_dir: Path,
) -> pd.DataFrame:
    seeds_a = set(runs_a["seed"].astype(str))
    seeds_b = set(runs_b["seed"].astype(str))
    if seeds_a != seeds_b:
        raise ValueError(f"Compared systems have different seeds: A={sorted(seeds_a)}, B={sorted(seeds_b)}")

    paired: list[pd.DataFrame] = []
    for seed in sorted(seeds_a):
        row_a = runs_a[runs_a["seed"].astype(str) == seed].iloc[0]
        row_b = runs_b[runs_b["seed"].astype(str) == seed].iloc[0]
        predictions_a = _load_prediction_file(row_a["prediction_file"], results_dir)
        predictions_b = _load_prediction_file(row_b["prediction_file"], results_dir)
        predictions_a = predictions_a[predictions_a["split"] == split]
        predictions_b = predictions_b[predictions_b["split"] == split]
        if predictions_a.empty or predictions_b.empty:
            raise ValueError(f"Prediction files have no split {split!r} for seed {seed}")
        merged = predictions_a.merge(
            predictions_b,
            on=["split", "example_id"],
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        if len(merged) != len(predictions_a) or len(merged) != len(predictions_b):
            raise ValueError(f"Compared systems cover different examples for seed {seed}")
        if not (merged["label_a"] == merged["label_b"]).all():
            raise ValueError(f"Compared systems disagree on labels for seed {seed}")
        if not (merged["question_id_a"].astype(str) == merged["question_id_b"].astype(str)).all():
            raise ValueError(f"Compared systems disagree on scenario groups for seed {seed}")
        merged["seed"] = seed
        paired.append(merged)
    return pd.concat(paired, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pre-registered paired hierarchical monitor comparisons"
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--comparisons", required=True, help="YAML with exact pre-registered systems")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = collect_results(args.results_dir)
    if df.empty:
        raise ValueError("No result summaries found")
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    config = yaml.safe_load(Path(args.comparisons).read_text(encoding="utf-8")) or {}
    comparisons = config.get("comparisons", [])
    if not comparisons:
        raise ValueError("Comparison file must define at least one pre-registered comparison")

    primary_selection_path = results_dir / "task_primary_source_systems.csv"
    primary_comparisons = [
        comparison
        for comparison in comparisons
        if comparison.get("comparison_role") == PRIMARY_COMPARISON_ROLE
    ]
    if primary_comparisons:
        if not primary_selection_path.exists():
            raise FileNotFoundError(
                f"Primary comparisons require {primary_selection_path}; run "
                "build_frozen_transfer_report first"
            )
        primary_selection = pd.read_csv(primary_selection_path)
    else:
        primary_selection = pd.DataFrame()

    output_rows: list[dict[str, Any]] = []
    for index, comparison in enumerate(comparisons):
        comparison_id = str(comparison.get("comparison_id", f"comparison_{index}"))
        common = comparison.get("common_filters", {})
        filters_a = {**common, **comparison.get("system_a", {})}
        filters_b = {**common, **comparison.get("system_b", {})}
        _validate_primary_source_selection(
            comparison,
            filters_a,
            filters_b,
            primary_selection,
            comparison_id,
        )
        runs_a = _filter_rows(df, filters_a, comparison_id)
        runs_b = _filter_rows(df, filters_b, comparison_id)
        split = comparison.get("split", "target_test")
        metric = comparison.get("metric", "tpr")
        paired = _paired_predictions(runs_a, runs_b, split, results_dir)
        result = hierarchical_paired_rate_difference(
            paired["label_a"].to_numpy(),
            paired["predicted_positive_a"].astype(float).to_numpy(),
            paired["predicted_positive_b"].astype(float).to_numpy(),
            paired["question_id_a"].astype(str).to_numpy(),
            paired["seed"].astype(str).to_numpy(),
            metric=metric,
            n_boot=args.bootstrap_samples,
            seed=args.seed + index,
        )
        output_rows.append(
            {
                "comparison_id": comparison_id,
                "description": comparison.get("description", ""),
                "split": split,
                "metric": metric,
                "system_a": json.dumps(filters_a, sort_keys=True),
                "system_b": json.dumps(filters_b, sort_keys=True),
                **result,
            }
        )

    adjusted = holm_adjust([float(row["p_value"]) for row in output_rows])
    for row, adjusted_p in zip(output_rows, adjusted):
        row["holm_adjusted_p_value"] = adjusted_p
    output = pd.DataFrame(output_rows)
    output_path = results_dir / "task_significance.csv"
    output.to_csv(output_path, index=False)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
