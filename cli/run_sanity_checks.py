from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final sanity checks on benchmark outputs")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--controls_report", default=None)
    parser.add_argument("--margin", type=float, default=0.02)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    calib = pd.read_csv(results_dir / "task_same_task_calibration.csv")
    transfer = pd.read_csv(results_dir / "task_cross_task_transfer.csv")

    checks = []
    same_mean = float(calib["test_recall_at_1pct_fpr_mean"].mean()) if not calib.empty else float("nan")
    transfer_mean = float(transfer["transfer_recall_at_1pct_fpr_mean"].mean()) if not transfer.empty else float("nan")
    checks.append({
        "check_name": "same_task_beats_cross_task",
        "passed": bool(same_mean >= transfer_mean),
        "value": same_mean - transfer_mean,
    })

    if args.controls_report:
        controls = pd.read_csv(args.controls_report)
        min_gap = float(controls["main_minus_control"].min()) if not controls.empty else float("nan")
        checks.append({
            "check_name": "main_beats_negative_controls",
            "passed": bool(min_gap >= args.margin),
            "value": min_gap,
        })

    signif_path = results_dir / "task_significance.csv"
    if signif_path.exists() and signif_path.stat().st_size > 0:
        try:
            signif = pd.read_csv(signif_path)
        except pd.errors.EmptyDataError:
            signif = pd.DataFrame()
        p_ok = float((signif["paired_permutation_pvalue"] < 0.05).mean()) if (not signif.empty and "paired_permutation_pvalue" in signif.columns) else 0.0
        checks.append({
            "check_name": "nontrivial_significance_fraction",
            "passed": bool(p_ok > 0.0),
            "value": p_ok,
        })

    steering_path = results_dir / "task_steering_best.csv"
    if steering_path.exists() and steering_path.stat().st_size > 0:
        try:
            steering = pd.read_csv(steering_path)
        except pd.errors.EmptyDataError:
            steering = pd.DataFrame()
        selectivity = float(steering["selectivity_score"].mean()) if (not steering.empty and "selectivity_score" in steering.columns) else float("nan")
        checks.append({
            "check_name": "steering_selectivity_positive",
            "passed": bool(selectivity > 0.0),
            "value": selectivity,
        })

    out_df = pd.DataFrame(checks)
    out_csv = results_dir / "sanity_checks.csv"
    out_json = results_dir / "sanity_checks.json"
    out_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(checks, indent=2))
    print(f"saved {out_csv}")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
