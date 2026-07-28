from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.text_views import parse_response_prefix_text_view
from data.trajectory_schema import (
    TRAJECTORY_PROMPT_END_VIEW,
    parse_trajectory_prefix_stack_view,
    parse_trajectory_prefix_view,
)


def _derive_prefix(view: str) -> int | None:
    if view == TRAJECTORY_PROMPT_END_VIEW:
        return 0
    percentile = parse_trajectory_prefix_view(view)
    if percentile is not None:
        return percentile
    percentile = parse_trajectory_prefix_stack_view(view)
    return percentile


def _trajectory_family(view: str) -> str | None:
    if view == TRAJECTORY_PROMPT_END_VIEW:
        return "prompt_control"
    if parse_trajectory_prefix_stack_view(view) is not None:
        return "stacked"
    if parse_trajectory_prefix_view(view) is not None:
        return "pooled"
    return None


def _integrate_curve(
    prefixes: pd.DataFrame, value_col: str, *, strategy: str = "early"
) -> float:
    if prefixes.empty or value_col not in prefixes.columns:
        return float("nan")
    if not {"prefix_alert_pct", value_col}.issubset(prefixes.columns):
        return float("nan")
    if strategy not in {"early", "uniform"}:
        raise ValueError(f"Unknown early-warning weighting strategy {strategy!r}")
    curve = prefixes[["prefix_alert_pct", value_col]].dropna().copy()
    if curve.empty:
        return float("nan")
    curve = curve.sort_values("prefix_alert_pct")
    x = curve["prefix_alert_pct"].to_numpy(dtype=float) / 100.0
    y = curve[value_col].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return float("nan")
    if strategy == "uniform":
        return float(np.trapezoid(y, x))

    # Early-emphasis weighting approximates a weighted-AUROC-like functional.
    weights = 1.0 - x
    weight_sum = float(np.trapezoid(weights, x))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return float("nan")
    return float(np.trapezoid(y * weights, x) / weight_sum)


EARLY_SELECTION_COLUMNS = [
    "model",
    "source_task",
    "k",
    "prefix_alert_pct",
    "white_probe",
    "white_balance_mode",
    "white_layer",
    "white_view",
    "black_probe",
    "black_balance_mode",
    "black_layer",
    "black_view",
    "selection_metric",
    "selection_rule",
]


def select_early_warning_systems(
    summary: pd.DataFrame,
    *,
    selection_k: int,
    expected_prefixes: set[int] | None = None,
) -> pd.DataFrame:
    """Select one P8 layer and one visible-text comparator per prefix.

    Selection uses only within-source evaluation rows. The P8 layer maximizes
    source-evaluation AUEW uplift over the strongest black-box system selected
    independently at each registered visible prefix.
    """

    required = {
        "model",
        "source_task",
        "target_task",
        "view",
        "probe",
        "k",
        "balance_mode",
        "layer",
        "eval_tpr_at_reference_alert_budget_mean",
    }
    if missing := required.difference(summary.columns):
        raise ValueError(
            f"Early-warning selection lacks columns: {sorted(missing)}"
        )
    source = summary[
        (summary["source_task"].astype(str) == summary["target_task"].astype(str))
        & (
            pd.to_numeric(summary["k"], errors="coerce")
            == int(selection_k)
        )
    ].copy()
    source["trajectory_prefix"] = source["view"].map(_derive_prefix)
    source["trajectory_family"] = source["view"].map(_trajectory_family)
    source["text_prefix"] = source["view"].map(parse_response_prefix_text_view)

    black = source[
        source["probe"].astype(str).str.startswith("B")
        & source["text_prefix"].notna()
    ].copy()
    if black.empty:
        raise ValueError(
            "Early-warning selection requires visible-prefix black-box rows"
        )
    black["prefix_alert_pct"] = black["text_prefix"].astype(int)
    black_group_keys = ["model", "source_task", "k", "prefix_alert_pct"]
    black = (
        black.sort_values(
            [
                *black_group_keys,
                "eval_tpr_at_reference_alert_budget_mean",
                "probe",
                "balance_mode",
                "layer",
                "view",
            ],
            ascending=[
                *([True] * len(black_group_keys)),
                False,
                True,
                True,
                True,
                True,
            ],
            kind="mergesort",
        )
        .groupby(black_group_keys, as_index=False, dropna=False)
        .head(1)
    )

    white = source[
        (source["probe"].astype(str) == "P8_citm")
        & (source["trajectory_family"].astype(str) == "stacked")
    ].copy()
    if white.empty:
        raise ValueError("Early-warning selection requires P8_citm stacked rows")
    white["prefix_alert_pct"] = white["trajectory_prefix"].astype(int)
    required_prefixes = expected_prefixes or set(
        white["prefix_alert_pct"].astype(int)
    )
    white_groups = {
        (str(model), str(source_task)): set(
            rows["prefix_alert_pct"].astype(int)
        )
        for (model, source_task), rows in white.groupby(
            ["model", "source_task"],
            dropna=False,
        )
    }
    black_groups = {
        (str(model), str(source_task)): set(
            rows["prefix_alert_pct"].astype(int)
        )
        for (model, source_task), rows in black.groupby(
            ["model", "source_task"],
            dropna=False,
        )
    }
    if set(white_groups) != set(black_groups):
        raise ValueError(
            "White- and black-box early-warning selection cover different "
            "model/source cells"
        )
    incomplete = {
        key: {
            "white": sorted(white_groups[key]),
            "black": sorted(black_groups[key]),
        }
        for key in white_groups
        if white_groups[key] != required_prefixes
        or black_groups[key] != required_prefixes
    }
    if incomplete:
        raise ValueError(
            "Early-warning selection does not exactly cover registered "
            f"prefixes {sorted(required_prefixes)}: {incomplete}"
        )

    black_identity = black[
        [
            *black_group_keys,
            "probe",
            "balance_mode",
            "layer",
            "view",
            "eval_tpr_at_reference_alert_budget_mean",
        ]
    ].rename(
        columns={
            "probe": "black_probe",
            "balance_mode": "black_balance_mode",
            "layer": "black_layer",
            "view": "black_view",
            "eval_tpr_at_reference_alert_budget_mean": "black_eval_tpr",
        }
    )
    candidates = white.merge(
        black_identity,
        on=black_group_keys,
        how="inner",
        validate="many_to_one",
    )
    candidates["eval_uplift"] = (
        candidates["eval_tpr_at_reference_alert_budget_mean"]
        - candidates["black_eval_tpr"]
    )
    candidate_keys = ["model", "source_task", "k", "balance_mode", "layer"]
    candidate_scores = (
        candidates.groupby(candidate_keys, dropna=False)
        .apply(
            lambda rows: _integrate_curve(
                rows,
                "eval_uplift",
                strategy="early",
            ),
            include_groups=False,
        )
        .rename("selection_metric")
        .reset_index()
    )
    if candidate_scores["selection_metric"].isna().any():
        raise ValueError(
            "Early-warning selection requires at least two finite prefix points "
            "for every P8 candidate"
        )
    winners = (
        candidate_scores.sort_values(
            [
                "model",
                "source_task",
                "selection_metric",
                "balance_mode",
                "layer",
            ],
            ascending=[True, True, False, True, True],
            kind="mergesort",
        )
        .groupby(["model", "source_task"], as_index=False, dropna=False)
        .head(1)
    )
    selected = candidates.merge(
        winners,
        on=candidate_keys,
        how="inner",
        validate="many_to_one",
    )
    selected = selected.rename(
        columns={
            "probe": "white_probe",
            "balance_mode": "white_balance_mode",
            "layer": "white_layer",
            "view": "white_view",
        }
    )
    selected["selection_rule"] = (
        "source_eval_max_early_weighted_auew_uplift_then_"
        "balance_layer_lexicographic_v1"
    )
    selected = selected[EARLY_SELECTION_COLUMNS].sort_values(
        ["model", "source_task", "prefix_alert_pct"],
        kind="mergesort",
    )
    return selected.reset_index(drop=True)


def load_frozen_early_selection(path: Path) -> pd.DataFrame:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("early_warning_selection")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"Frozen comparison file {path} lacks early_warning_selection"
        )
    selected = pd.DataFrame(rows)
    if missing := set(EARLY_SELECTION_COLUMNS).difference(selected.columns):
        raise ValueError(
            f"Frozen early-warning selection lacks columns {sorted(missing)}"
        )
    return selected[EARLY_SELECTION_COLUMNS].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an early-warning monitoring report from activation trajectory "
            "prefix views."
        )
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--selection_k", type=int, default=8)
    parser.add_argument(
        "--frozen_comparisons",
        help="Frozen comparison YAML containing source-only early_warning_selection.",
    )
    args = parser.parse_args()

    summary_path = Path(args.results_dir) / "task_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}; run aggregate_task_results.py first.")

    summary = pd.read_csv(summary_path)
    required = {"model", "source_task", "target_task", "view", "probe", "k", "balance_mode", "layer"}
    if missing := required.difference(summary.columns):
        raise ValueError(f"task_summary.csv lacks required columns: {sorted(missing)}")

    summary["trajectory_prefix"] = summary["view"].map(_derive_prefix)
    summary["trajectory_family"] = summary["view"].map(_trajectory_family)
    summary["text_prefix"] = summary["view"].map(parse_response_prefix_text_view)
    answer_rows = summary[
        summary["view"].astype(str).str.casefold() == "answer"
    ].copy()
    trajectory_rows = summary[summary["trajectory_prefix"].notna()].copy()
    if trajectory_rows.empty:
        print("No trajectory_prefix_* view rows found in task_summary.csv; skipping.")
        return
    if args.frozen_comparisons:
        early_selection = load_frozen_early_selection(
            Path(args.frozen_comparisons)
        )
    else:
        early_selection = select_early_warning_systems(
            summary,
            selection_k=int(args.selection_k),
        )
    early_selection.to_csv(
        Path(args.results_dir) / "early_warning_source_selection.csv",
        index=False,
    )

    merge_keys = ["model", "source_task", "target_task", "probe", "k", "balance_mode", "layer"]
    answer_cols = [
        "eval_tpr_at_reference_alert_budget_mean",
        "eval_tpr_at_1pct_reference_alert_budget_mean",
        "transfer_tpr_at_reference_alert_budget_mean",
        "transfer_tpr_at_1pct_reference_alert_budget_mean",
    ]
    missing_cols = [col for col in ["eval_tpr_at_reference_alert_budget_mean", "transfer_tpr_at_reference_alert_budget_mean"] if col not in summary.columns]
    if missing_cols:
        raise ValueError(
            f"task_summary.csv is not an aggregated result file; missing {missing_cols}"
        )
    for col in answer_cols:
        if col not in summary.columns:
            answer_rows[col] = float("nan")
            trajectory_rows[col] = float("nan")

    merged = trajectory_rows.merge(
        answer_rows[merge_keys + answer_cols],
        on=merge_keys,
        suffixes=("", "_answer"),
        how="left",
        validate="many_to_one",
    )
    black_prefix_rows = summary[
        summary["probe"].astype(str).str.startswith("B")
        & summary["text_prefix"].notna()
    ].copy()
    if black_prefix_rows.empty:
        raise ValueError(
            "Early-warning report requires registered visible-prefix black-box rows"
        )
    black_prefix_rows["prefix_alert_pct"] = black_prefix_rows[
        "text_prefix"
    ].astype(int)
    selected_black = black_prefix_rows.merge(
        early_selection,
        left_on=[
            "model",
            "source_task",
            "k",
            "prefix_alert_pct",
            "probe",
            "balance_mode",
            "layer",
            "view",
        ],
        right_on=[
            "model",
            "source_task",
            "k",
            "prefix_alert_pct",
            "black_probe",
            "black_balance_mode",
            "black_layer",
            "black_view",
        ],
        how="inner",
        validate="many_to_one",
    )
    selected_black = selected_black.rename(
        columns={
            "probe": "matched_black_probe",
            "view": "matched_black_view",
            "balance_mode": "matched_black_balance_mode",
            "eval_tpr_at_reference_alert_budget_mean": (
                "matched_black_eval_tpr_at_reference_alert_budget_mean"
            ),
            "transfer_tpr_at_reference_alert_budget_mean": (
                "matched_black_transfer_tpr_at_reference_alert_budget_mean"
            ),
        }
    )
    black_group_keys = [
        "model",
        "source_task",
        "target_task",
        "k",
        "prefix_alert_pct",
    ]
    merged["prefix_alert_pct"] = merged["trajectory_prefix"].astype(int)
    merged = merged.merge(
        selected_black[
            black_group_keys
            + [
                "matched_black_probe",
                "matched_black_view",
                "matched_black_balance_mode",
                "matched_black_eval_tpr_at_reference_alert_budget_mean",
                "matched_black_transfer_tpr_at_reference_alert_budget_mean",
            ]
        ],
        on=black_group_keys,
        how="left",
        validate="many_to_one",
    )
    merged["transfer_tpr_delta_vs_answer"] = (
        merged["transfer_tpr_at_reference_alert_budget_mean"]
        - merged["transfer_tpr_at_reference_alert_budget_mean_answer"]
    )
    merged["eval_tpr_delta_vs_answer"] = (
        merged["eval_tpr_at_reference_alert_budget_mean"]
        - merged["eval_tpr_at_reference_alert_budget_mean_answer"]
    )
    merged["transfer_1pct_tpr_delta_vs_answer"] = (
        merged["transfer_tpr_at_1pct_reference_alert_budget_mean"]
        - merged["transfer_tpr_at_1pct_reference_alert_budget_mean_answer"]
    )
    merged["eval_1pct_tpr_delta_vs_answer"] = (
        merged["eval_tpr_at_1pct_reference_alert_budget_mean"]
        - merged["eval_tpr_at_1pct_reference_alert_budget_mean_answer"]
    )
    merged["eval_tpr_uplift_vs_matched_black"] = (
        merged["eval_tpr_at_reference_alert_budget_mean"]
        - merged["matched_black_eval_tpr_at_reference_alert_budget_mean"]
    )
    merged["transfer_tpr_uplift_vs_matched_black"] = (
        merged["transfer_tpr_at_reference_alert_budget_mean"]
        - merged["matched_black_transfer_tpr_at_reference_alert_budget_mean"]
    )
    curve_group_keys = [
        "model",
        "source_task",
        "target_task",
        "probe",
        "k",
        "balance_mode",
        "layer",
        "trajectory_family",
    ]
    merged["auew_transfer_tpr_delta_vs_answer"] = merged.groupby(
        curve_group_keys,
        dropna=False,
    )["transfer_tpr_delta_vs_answer"].transform(
        lambda values: _integrate_curve(
            merged.loc[values.index],
            "transfer_tpr_delta_vs_answer",
            strategy="early",
        )
    )
    merged["auew_eval_tpr_delta_vs_answer"] = merged.groupby(
        curve_group_keys,
        dropna=False,
    )["eval_tpr_delta_vs_answer"].transform(
        lambda values: _integrate_curve(
            merged.loc[values.index],
            "eval_tpr_delta_vs_answer",
            strategy="early",
        )
    )
    merged["auew_transfer_tpr"] = merged.groupby(
        curve_group_keys,
        dropna=False,
    )["transfer_tpr_at_reference_alert_budget_mean"].transform(
        lambda values: _integrate_curve(
            merged.loc[values.index],
            "transfer_tpr_at_reference_alert_budget_mean",
            strategy="early",
        )
    )
    merged["auew_eval_tpr"] = merged.groupby(
        curve_group_keys,
        dropna=False,
    )["eval_tpr_at_reference_alert_budget_mean"].transform(
        lambda values: _integrate_curve(
            merged.loc[values.index],
            "eval_tpr_at_reference_alert_budget_mean",
            strategy="early",
        )
    )
    merged["auew_eval_uplift_vs_matched_black"] = merged.groupby(
        curve_group_keys,
        dropna=False,
    )["eval_tpr_uplift_vs_matched_black"].transform(
        lambda values: _integrate_curve(
            merged.loc[values.index],
            "eval_tpr_uplift_vs_matched_black",
            strategy="early",
        )
    )
    merged["auew_transfer_uplift_vs_matched_black"] = merged.groupby(
        curve_group_keys,
        dropna=False,
    )["transfer_tpr_uplift_vs_matched_black"].transform(
        lambda values: _integrate_curve(
            merged.loc[values.index],
            "transfer_tpr_uplift_vs_matched_black",
            strategy="early",
        )
    )

    columns = [
        "model",
        "source_task",
        "target_task",
        "probe",
        "k",
        "balance_mode",
        "layer",
        "view",
        "trajectory_family",
        "trajectory_prefix",
        "prefix_alert_pct",
        "matched_black_probe",
        "matched_black_view",
        "matched_black_balance_mode",
        "eval_tpr_at_reference_alert_budget_mean",
        "eval_tpr_at_reference_alert_budget_mean_answer",
        "eval_tpr_delta_vs_answer",
        "eval_tpr_at_1pct_reference_alert_budget_mean",
        "eval_tpr_at_1pct_reference_alert_budget_mean_answer",
        "eval_1pct_tpr_delta_vs_answer",
        "auew_eval_tpr_delta_vs_answer",
        "auew_eval_tpr",
        "matched_black_eval_tpr_at_reference_alert_budget_mean",
        "eval_tpr_uplift_vs_matched_black",
        "auew_eval_uplift_vs_matched_black",
        "transfer_tpr_at_reference_alert_budget_mean",
        "transfer_tpr_at_reference_alert_budget_mean_answer",
        "transfer_tpr_delta_vs_answer",
        "transfer_tpr_at_1pct_reference_alert_budget_mean",
        "transfer_tpr_at_1pct_reference_alert_budget_mean_answer",
        "transfer_1pct_tpr_delta_vs_answer",
        "auew_transfer_tpr_delta_vs_answer",
        "auew_transfer_tpr",
        "matched_black_transfer_tpr_at_reference_alert_budget_mean",
        "transfer_tpr_uplift_vs_matched_black",
        "auew_transfer_uplift_vs_matched_black",
    ]
    merged = merged[columns]

    outdir = Path(args.results_dir)
    merged.sort_values(
        ["model", "source_task", "target_task", "probe", "k", "balance_mode", "trajectory_prefix"],
        inplace=True,
    )
    merged.to_csv(outdir / "early_warning_report.csv", index=False)

    auew = (
        merged[
            curve_group_keys
            + [
                "auew_eval_tpr_delta_vs_answer",
                "auew_transfer_tpr_delta_vs_answer",
                "auew_eval_tpr",
                "auew_transfer_tpr",
                "auew_eval_uplift_vs_matched_black",
                "auew_transfer_uplift_vs_matched_black",
            ]
        ]
        .drop_duplicates()
        .sort_values(curve_group_keys)
    )
    auew_path = outdir / "early_warning_auew.csv"
    auew.to_csv(auew_path, index=False)

    citm = auew[
        (auew["probe"].astype(str) == "P8_citm")
        & (pd.to_numeric(auew["k"], errors="coerce") == int(args.selection_k))
        & (auew["trajectory_family"].astype(str) == "stacked")
    ].copy()
    selected_white = early_selection[
        [
            "model",
            "source_task",
            "k",
            "white_probe",
            "white_balance_mode",
            "white_layer",
        ]
    ].drop_duplicates()
    primary_details = citm.merge(
        selected_white,
        left_on=[
            "model",
            "source_task",
            "k",
            "probe",
            "balance_mode",
            "layer",
        ],
        right_on=[
            "model",
            "source_task",
            "k",
            "white_probe",
            "white_balance_mode",
            "white_layer",
        ],
        how="inner",
        validate="many_to_one",
    )
    primary_details = primary_details[
        primary_details["source_task"].astype(str)
        != primary_details["target_task"].astype(str)
    ].copy()
    primary_details_path = outdir / "early_warning_primary_details.csv"
    primary_details.to_csv(primary_details_path, index=False)
    finite_primary = pd.to_numeric(
        primary_details.get(
            "auew_transfer_uplift_vs_matched_black",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    ).dropna()
    endpoint = pd.DataFrame(
        [
            {
                "endpoint": "mean_unseen_behavior_auew_uplift",
                "probe": "P8_citm",
                "selection_k": int(args.selection_k),
                "reference_alert_budget": 0.01,
                "n_model_target_cells": int(len(finite_primary)),
                "estimate": (
                    float(finite_primary.mean())
                    if len(finite_primary)
                    else float("nan")
                ),
                "status": (
                    "descriptive_pending_hierarchical_inference"
                    if len(finite_primary)
                    else "unavailable_until_confirmatory_execution"
                ),
            }
        ]
    )
    endpoint_path = outdir / "early_warning_primary_endpoint.csv"
    endpoint.to_csv(endpoint_path, index=False)

    print(f"saved {outdir / 'early_warning_report.csv'}")
    print(f"saved {auew_path}")
    print(f"saved {primary_details_path}")
    print(f"saved {endpoint_path}")


if __name__ == "__main__":
    main()
