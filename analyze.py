"""Analyze probe benchmark outputs and save paper ready tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from evaluation.aggregation import (
    collect_results,
    compute_fsei_table,
    compute_summary_stats,
    make_decision_table,
    make_layer_choices,
    make_ood_table,
    select_best_layer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze probe sweep results")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir = args.results_dir or config["results_dir"]
    k_values = config["sweep"]["k_values"]
    selection_metric = config.get("selection_metric", "eval_recall_at_1pct_fpr")

    metric_col = f"{selection_metric}_mean"
    if metric_col not in {
        "eval_auroc_mean",
        "eval_recall_at_1pct_fpr_mean",
        "test_auroc_mean",
        "test_recall_at_1pct_fpr_mean",
    }:
        metric_col = "eval_recall_at_1pct_fpr_mean"

    print("Loading results")
    df = collect_results(results_dir)
    print(f"  found {len(df)} rows")

    if df.empty:
        print("No results found")
        return

    print("Computing summary statistics")
    summary = compute_summary_stats(df)

    print(f"Selecting best layer using {metric_col}")
    best = select_best_layer(summary, selection_metric=metric_col)

    print("Computing FSEI")
    fsei = compute_fsei_table(best, k_values)

    print("Building decision table")
    decision = make_decision_table(best)

    print("Building OOD table")
    ood_table = make_ood_table(best)

    print("Saving layer choice audit")
    layer_choices = make_layer_choices(best)

    out = Path(results_dir)
    summary.to_csv(out / "summary.csv", index=False)
    best.to_csv(out / "best_layer_summary.csv", index=False)
    fsei.to_csv(out / "fsei.csv", index=False)
    decision.to_csv(out / "decision_table.csv", index=False)
    ood_table.to_csv(out / "ood_table.csv", index=False)
    layer_choices.to_csv(out / "layer_choices.csv", index=False)

    print(f"\nSaved analysis artifacts to {out}/")
    print(f"  summary.csv: {len(summary)} rows")
    print(f"  best_layer_summary.csv: {len(best)} rows")
    print(f"  fsei.csv: {len(fsei)} rows")
    print(f"  decision_table.csv: {len(decision)} rows")
    print(f"  ood_table.csv: {len(ood_table)} rows")
    print(f"  layer_choices.csv: {len(layer_choices)} rows")

    if not decision.empty:
        print("\nDecision table preview")
        print(decision.head(20).to_string(index=False))

    if not fsei.empty:
        print("\nFSEI ranking preview")
        print(fsei.sort_values("fsei", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
