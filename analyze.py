"""Aggregate sweep results and compute summary statistics, FSEI, and decision table.

Usage:
    python analyze.py --config config.yaml
    python analyze.py --config config.yaml --results-dir results/
"""

import argparse
from pathlib import Path

import yaml

from evaluation.aggregation import (
    collect_results,
    compute_summary_stats,
    select_best_layer,
    compute_fsei_table,
    make_decision_table,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze probe sweep results")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = args.results_dir or config["results_dir"]
    k_values = config["sweep"]["k_values"]

    print("Loading results...")
    df = collect_results(results_dir)
    print(f"  {len(df)} total experiment rows")

    if df.empty:
        print("No results found. Run the sweep first.")
        return

    print("\nComputing summary statistics...")
    summary = compute_summary_stats(df)

    print("Selecting best layers...")
    best = select_best_layer(summary)

    print("Computing FSEI...")
    fsei = compute_fsei_table(df, k_values)

    print("Building decision table...")
    decision = make_decision_table(best, k_values)

    # Save outputs
    out = Path(results_dir)
    summary.to_csv(out / "summary.csv", index=False)
    best.to_csv(out / "best_layer_summary.csv", index=False)
    fsei.to_csv(out / "fsei.csv", index=False)
    decision.to_csv(out / "decision_table.csv", index=False)

    print(f"\nOutputs saved to {out}/")
    print(f"  summary.csv          ({len(summary)} rows)")
    print(f"  best_layer_summary.csv ({len(best)} rows)")
    print(f"  fsei.csv             ({len(fsei)} rows)")
    print(f"  decision_table.csv   ({len(decision)} rows)")

    # Print the decision table
    print("\n" + "=" * 60)
    print("DECISION TABLE: Best probe by k")
    print("=" * 60)
    if not decision.empty:
        print(decision.to_string(index=False))

    # Print FSEI rankings
    print("\n" + "=" * 60)
    print("FSEI RANKINGS")
    print("=" * 60)
    if not fsei.empty:
        ranked = fsei.sort_values("fsei", ascending=False)
        print(ranked.to_string(index=False))


if __name__ == "__main__":
    main()
