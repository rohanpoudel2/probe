from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate only pre-registered paired claims from scenario-level inference"
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--controls_report", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    significance_path = results_dir / "task_significance.csv"
    if not significance_path.exists():
        raise FileNotFoundError(
            f"Missing {significance_path}; claims require a pre-registered comparisons file and per-example inference"
        )
    significance = pd.read_csv(significance_path)
    required = {
        "comparison_id",
        "description",
        "mean_diff",
        "ci_low",
        "ci_high",
        "holm_adjusted_p_value",
        "n_groups",
        "n_seeds",
    }
    missing = sorted(required.difference(significance.columns))
    if missing:
        raise ValueError(f"Significance artifact is missing required columns: {missing}")

    rows = []
    for comparison in significance.itertuples(index=False):
        passed = bool(
            float(comparison.ci_low) > 0.0
            and float(comparison.holm_adjusted_p_value) < args.alpha
        )
        rows.append(
            {
                "claim_id": comparison.comparison_id,
                "claim": comparison.description,
                "estimate": float(comparison.mean_diff),
                "ci_low": float(comparison.ci_low),
                "ci_high": float(comparison.ci_high),
                "holm_adjusted_p_value": float(comparison.holm_adjusted_p_value),
                "alpha": args.alpha,
                "n_groups": int(comparison.n_groups),
                "n_seeds": int(comparison.n_seeds),
                "passed": passed,
                "decision_rule": "paired difference CI excludes zero and Holm-adjusted p < alpha",
            }
        )

    if args.controls_report:
        controls_path = Path(args.controls_report)
        if not controls_path.exists():
            raise FileNotFoundError(f"Missing declared controls report: {controls_path}")
        controls = pd.read_csv(controls_path)
        if "main_minus_control" not in controls:
            raise ValueError("Controls report must contain main_minus_control")
        rows.append(
            {
                "claim_id": "registered_negative_controls",
                "claim": "Every pre-registered negative control underperforms the primary monitor",
                "estimate": float(controls["main_minus_control"].min()),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "holm_adjusted_p_value": float("nan"),
                "alpha": args.alpha,
                "n_groups": 0,
                "n_seeds": 0,
                "passed": bool((controls["main_minus_control"] > 0.0).all()),
                "decision_rule": "all registered control gaps > 0; descriptive until paired CIs are supplied",
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = results_dir / "claim_tests.csv"
    out_json = results_dir / "claim_tests.json"
    out_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {out_csv}")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
