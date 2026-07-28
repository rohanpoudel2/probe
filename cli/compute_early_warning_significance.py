from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cli.build_early_warning_report import load_frozen_early_selection
from cli.compute_task_significance import (
    _archive_comparisons,
    _load_prediction_file,
)
from evaluation.aggregation import collect_results
from evaluation.hierarchical_statistics import (
    hierarchical_paired_curve_aggregate,
)


def _filter_runs(
    runs: pd.DataFrame,
    filters: dict[str, Any],
    *,
    identity: str,
) -> pd.DataFrame:
    selected = runs
    for key, value in filters.items():
        if key not in selected.columns:
            raise ValueError(
                f"Early-warning {identity} filters on absent column {key!r}"
            )
        if key in {"k", "layer"}:
            mask = pd.to_numeric(selected[key], errors="coerce") == float(value)
        else:
            mask = selected[key].astype(str) == str(value)
        selected = selected[mask]
    if selected.empty:
        raise ValueError(
            f"Early-warning {identity} selected no runs for {filters}"
        )
    if selected["seed"].astype(str).duplicated(keep=False).any():
        raise ValueError(
            f"Early-warning {identity} does not identify one run per seed"
        )
    return selected.copy()


def _paired_prefix_predictions(
    runs_a: pd.DataFrame,
    runs_b: pd.DataFrame,
    *,
    results_dir: Path,
    prefix: int,
) -> pd.DataFrame:
    seeds_a = set(runs_a["seed"].astype(str))
    seeds_b = set(runs_b["seed"].astype(str))
    if seeds_a != seeds_b:
        raise ValueError(
            "Early-warning systems have different seed coverage: "
            f"A={sorted(seeds_a)}, B={sorted(seeds_b)}"
        )
    paired: list[pd.DataFrame] = []
    for seed in sorted(seeds_a):
        row_a = runs_a[runs_a["seed"].astype(str) == seed].iloc[0]
        row_b = runs_b[runs_b["seed"].astype(str) == seed].iloc[0]
        predictions_a = _load_prediction_file(
            row_a["prediction_file"],
            results_dir,
        )
        predictions_b = _load_prediction_file(
            row_b["prediction_file"],
            results_dir,
        )
        predictions_a = predictions_a[
            predictions_a["split"].astype(str) == "target_test"
        ]
        predictions_b = predictions_b[
            predictions_b["split"].astype(str) == "target_test"
        ]
        merged = predictions_a.merge(
            predictions_b,
            on=["split", "example_id"],
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        if (
            merged.empty
            or len(merged) != len(predictions_a)
            or len(merged) != len(predictions_b)
        ):
            raise ValueError(
                f"Early-warning predictions do not pair at prefix {prefix}"
            )
        if not (merged["label_a"] == merged["label_b"]).all():
            raise ValueError(
                f"Early-warning labels disagree at prefix {prefix}"
            )
        if not (
            merged["question_id_a"].astype(str)
            == merged["question_id_b"].astype(str)
        ).all():
            raise ValueError(
                f"Early-warning scenario groups disagree at prefix {prefix}"
            )
        merged = merged[merged["label_a"].astype(int) == 1].copy()
        if merged.empty:
            raise ValueError(
                f"Early-warning TPR has no positive examples at prefix {prefix}"
            )
        merged["seed"] = seed
        merged["prefix_alert_pct"] = int(prefix)
        paired.append(merged)
    return pd.concat(paired, ignore_index=True)


def _registered_unseen_cells(
    comparisons: dict[str, Any],
    *,
    selection_k: int,
) -> list[tuple[str, str, str]]:
    cells = {
        (
            str(common["model"]),
            str(common["source_task"]),
            str(common["target_task"]),
        )
        for comparison in comparisons.get("comparisons", [])
        if comparison.get("comparison_role") == "primary_white_box_gain"
        for common in [comparison.get("common_filters", {})]
        if int(common.get("k", -1)) == int(selection_k)
        and str(common.get("source_task")) != str(common.get("target_task"))
    }
    if not cells:
        raise ValueError(
            "Frozen comparisons define no unseen-behavior cells at selection_k"
        )
    return sorted(cells)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the frozen hierarchical AUEW uplift endpoint from "
            "per-example prefix predictions."
        )
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--comparisons", required=True)
    parser.add_argument("--selection_k", type=int, default=8)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    comparison_path = Path(args.comparisons)
    comparisons = (
        yaml.safe_load(comparison_path.read_text(encoding="utf-8")) or {}
    )
    selection = load_frozen_early_selection(comparison_path)
    if set(pd.to_numeric(selection["k"], errors="raise").astype(int)) != {
        int(args.selection_k)
    }:
        raise ValueError(
            "Frozen early-warning selection disagrees with --selection_k"
        )
    comparisons_sha256 = _archive_comparisons(
        results_dir,
        comparison_path,
    )
    runs = collect_results(str(results_dir))
    if runs.empty:
        raise ValueError("No result summaries found")
    runs = runs[runs["status"].astype(str) == "ok"].copy()

    cells: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for cell_index, (model, source_task, target_task) in enumerate(
        _registered_unseen_cells(
            comparisons,
            selection_k=int(args.selection_k),
        )
    ):
        selectors = selection[
            (selection["model"].astype(str) == model)
            & (selection["source_task"].astype(str) == source_task)
        ].sort_values("prefix_alert_pct")
        if selectors.empty:
            raise ValueError(
                f"No frozen early-warning selector for {model}/{source_task}"
            )
        paired_prefixes: list[pd.DataFrame] = []
        for selector in selectors.itertuples(index=False):
            common = {
                "model": model,
                "source_task": source_task,
                "target_task": target_task,
                "k": int(args.selection_k),
            }
            white_filters = {
                **common,
                "probe": selector.white_probe,
                "balance_mode": selector.white_balance_mode,
                "layer": selector.white_layer,
                "view": selector.white_view,
            }
            black_filters = {
                **common,
                "probe": selector.black_probe,
                "balance_mode": selector.black_balance_mode,
                "layer": selector.black_layer,
                "view": selector.black_view,
            }
            white_runs = _filter_runs(
                runs,
                white_filters,
                identity="white system",
            )
            black_runs = _filter_runs(
                runs,
                black_filters,
                identity="black system",
            )
            paired_prefixes.append(
                _paired_prefix_predictions(
                    white_runs,
                    black_runs,
                    results_dir=results_dir,
                    prefix=int(selector.prefix_alert_pct),
                )
            )
        paired = pd.concat(paired_prefixes, ignore_index=True)
        cell = {
            "values_a": paired["predicted_positive_a"].astype(float).to_numpy(),
            "values_b": paired["predicted_positive_b"].astype(float).to_numpy(),
            "group_ids": paired["question_id_a"].astype(str).to_numpy(),
            "seed_ids": paired["seed"].astype(str).to_numpy(),
            "prefix_ids": paired["prefix_alert_pct"].astype(int).to_numpy(),
        }
        cells.append(cell)
        cell_result = hierarchical_paired_curve_aggregate(
            [cell],
            n_boot=int(args.bootstrap_samples),
            seed=int(args.seed) + cell_index + 1,
        )
        cell_rows.append(
            {
                "model": model,
                "source_task": source_task,
                "target_task": target_task,
                "selection_k": int(args.selection_k),
                **cell_result,
            }
        )

    aggregate = hierarchical_paired_curve_aggregate(
        cells,
        n_boot=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    aggregate_row = {
        "endpoint": "mean_unseen_behavior_auew_uplift",
        "probe": "P8_citm",
        "selection_k": int(args.selection_k),
        "reference_alert_budget": 0.01,
        "comparisons_sha256": comparisons_sha256,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.seed),
        "status": "confirmatory_hierarchical_inference_complete",
        **aggregate,
    }
    pd.DataFrame(cell_rows).to_csv(
        results_dir / "early_warning_cell_inference.csv",
        index=False,
    )
    pd.DataFrame([aggregate_row]).to_csv(
        results_dir / "early_warning_primary_inference.csv",
        index=False,
    )
    (results_dir / "early_warning_primary_inference.json").write_text(
        json.dumps(aggregate_row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"saved {results_dir / 'early_warning_primary_inference.csv'}"
    )


if __name__ == "__main__":
    main()
