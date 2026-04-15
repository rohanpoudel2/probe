from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run claim-aligned summary tests for the paper")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--controls_report", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    calibration = pd.read_csv(results_dir / "task_same_task_calibration.csv")
    transfer = pd.read_csv(results_dir / "task_cross_task_transfer.csv")
    frozen = pd.read_csv(results_dir / "task_frozen_transfer_report.csv")

    rows = []

    same_mean = float(calibration["test_recall_at_1pct_fpr_mean"].mean()) if not calibration.empty else float("nan")
    cross_mean = float(transfer["transfer_recall_at_1pct_fpr_mean"].mean()) if not transfer.empty else float("nan")
    rows.append({
        "claim_id": "C1",
        "claim": "same-task calibration exceeds cross-task transfer",
        "value": same_mean - cross_mean,
        "passed": bool(same_mean >= cross_mean),
    })

    by_model = transfer.groupby("model", dropna=False)["transfer_recall_at_1pct_fpr_mean"].mean().reset_index()
    positive_frac = float((by_model["transfer_recall_at_1pct_fpr_mean"] > 0.5).mean()) if not by_model.empty else 0.0
    rows.append({
        "claim_id": "C2",
        "claim": "transfer remains above chance across models",
        "value": positive_frac,
        "passed": bool(positive_frac >= 0.5),
    })

    frozen_gap = frozen.groupby(["model", "source_task"], dropna=False)["transfer_recall_at_1pct_fpr_mean"].mean().reset_index()
    stable_frac = float((frozen_gap["transfer_recall_at_1pct_fpr_mean"] > 0.5).mean()) if not frozen_gap.empty else 0.0
    rows.append({
        "claim_id": "C3",
        "claim": "frozen-source transfer stays nontrivial across model-source groups",
        "value": stable_frac,
        "passed": bool(stable_frac >= 0.5),
    })

    if args.controls_report and Path(args.controls_report).exists() and Path(args.controls_report).stat().st_size > 0:
        controls = pd.read_csv(args.controls_report)
        min_gap = float(controls["main_minus_control"].min()) if not controls.empty else float("nan")
        rows.append({
            "claim_id": "C4",
            "claim": "main results beat negative controls",
            "value": min_gap,
            "passed": bool(min_gap > 0.0),
        })

    steering_path = results_dir / "task_steering_best.csv"
    if steering_path.exists() and steering_path.stat().st_size > 0:
        steering = pd.read_csv(steering_path)
        steering = steering[steering["steering_mode"] == "threshold_triggered"] if "steering_mode" in steering.columns else steering
        selectivity = float(steering["selectivity_score"].mean()) if not steering.empty and "selectivity_score" in steering.columns else float("nan")
        rows.append({
            "claim_id": "C5",
            "claim": "threshold-triggered steering is selective",
            "value": selectivity,
            "passed": bool(selectivity > 0.0),
        })

    out_df = pd.DataFrame(rows)
    out_csv = results_dir / "claim_tests.csv"
    out_json = results_dir / "claim_tests.json"
    out_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"saved {out_csv}")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
