"""Run paired bootstrap significance tests from saved benchmark artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from evaluation.significance import build_pairwise_significance_table, compare_two_runs


def _compare_pair(
    predictions_dir: Path,
    row: pd.Series,
    split: str,
    metric_name: str,
    n_boot: int,
    seed: int,
) -> dict:
    result = compare_two_runs(
        predictions_dir=predictions_dir,
        run_id_a=row["run_id_a"],
        run_id_b=row["run_id_b"],
        split=split,
        metric_name=metric_name,
        n_boot=n_boot,
        seed=seed,
    )
    result.update(
        {
            "dataset": row["dataset"],
            "model": row["model"],
            "balance_mode": row["balance_mode"],
            "k": row["k"],
            "probe_a": row["probe_a"],
            "probe_b": row["probe_b"],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run significance testing for benchmark outputs")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--split", default="test", choices=["eval", "test"])
    parser.add_argument(
        "--metric",
        default="recall_at_1pct_fpr",
        choices=["recall_at_1pct_fpr", "auroc"],
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir = Path(args.results_dir or config.get("results_dir", "./results"))
    predictions_dir = results_dir / "predictions"
    n_boot = int(config.get("significance", {}).get("n_boot", 2000))
    seed = int(config.get("significance", {}).get("seed", 0))

    decision_path = results_dir / "decision_table.csv"
    summary_path = results_dir / "best_layer_summary.csv"
    raw_results = []
    for jsonl_path in sorted(results_dir.glob("*.jsonl")):
        raw_results.append(pd.read_json(jsonl_path, lines=True))
    full_results = pd.concat(raw_results, ignore_index=True) if raw_results else pd.DataFrame()

    if not decision_path.exists() or not summary_path.exists() or full_results.empty:
        raise SystemExit("Missing analysis outputs or raw result rows. Run sweep and analyze first.")

    decision = pd.read_csv(decision_path)
    pairwise = build_pairwise_significance_table(decision_table=decision, full_results=full_results)

    rows = []
    for _, row in pairwise.iterrows():
        try:
            rows.append(
                _compare_pair(
                    predictions_dir=predictions_dir,
                    row=row,
                    split=args.split,
                    metric_name=args.metric,
                    n_boot=n_boot,
                    seed=seed,
                )
            )
        except Exception as err:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "balance_mode": row["balance_mode"],
                    "k": row["k"],
                    "probe_a": row["probe_a"],
                    "probe_b": row["probe_b"],
                    "run_id_a": row["run_id_a"],
                    "run_id_b": row["run_id_b"],
                    "split": args.split,
                    "metric": args.metric,
                    "error": type(err).__name__,
                }
            )

    out_df = pd.DataFrame(rows)
    out_path = results_dir / f"significance_{args.split}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved significance results to {out_path}")


if __name__ == "__main__":
    main()
