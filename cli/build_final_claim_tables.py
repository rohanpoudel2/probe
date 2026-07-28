from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cli.common import load_yaml
from evaluation.aggregation import collect_results


def _to_tex(df: pd.DataFrame, path: Path) -> None:
    path.write_text(
        df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}"),
        encoding="utf-8",
    )


def _reference_budget_status(
    low: pd.Series,
    high: pd.Series,
    *,
    budget: float,
) -> pd.Series:
    status = pd.Series("inconclusive", index=low.index, dtype="string")
    status.loc[high <= budget] = "supported"
    status.loc[low > budget] = "violated"
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create frozen primary and supporting claim tables"
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    config = load_yaml(args.config)
    claim_gates = config.get("claim_gates") or {}
    raw_results = collect_results(results_dir)
    if raw_results.empty:
        raise ValueError("No result rows are available for claim tables")
    required_reference_fields = {
        "reference_holdout_alert_rate",
        "reference_holdout_alert_rate_ci_low",
        "reference_holdout_alert_rate_ci_high",
        "reference_holdout_alert_budget_violation",
    }
    missing_reference_fields = sorted(
        required_reference_fields.difference(raw_results.columns)
    )
    if missing_reference_fields:
        raise ValueError(
            "Claim tables require held-out reference evidence; missing "
            f"{missing_reference_fields}"
        )
    if raw_results[list(required_reference_fields)].isna().any().any():
        raise ValueError(
            "Claim tables require held-out reference evidence on every run"
        )
    reference_budget = float(config["max_reference_alert_rate"])
    expected_violation = (
        pd.to_numeric(
            raw_results["reference_holdout_alert_rate_ci_low"], errors="raise"
        )
        > reference_budget
    )
    recorded_violation = raw_results[
        "reference_holdout_alert_budget_violation"
    ].astype(bool)
    if not expected_violation.equals(recorded_violation):
        raise ValueError(
            "Held-out reference alert-budget diagnostics are inconsistent with "
            "their Wilson intervals"
        )
    falsification_path = results_dir / "falsification_significance.csv"
    if not falsification_path.exists():
        raise FileNotFoundError(
            "Claim tables require falsification_significance.csv"
        )
    falsification = pd.read_csv(falsification_path)
    hard_fpr = falsification[falsification["metric"] == "hard_negative_fpr"]
    ordering = falsification[
        falsification["metric"] == "pairwise_order_accuracy"
    ]
    if hard_fpr.empty or ordering.empty:
        raise ValueError(
            "Claim tables require registered hard-negative FPR and pairwise-ordering "
            "comparisons"
        )
    noninferiority_margin = float(
        claim_gates["hard_negative_fpr_noninferiority_margin"]
    )
    ordering_floor = float(claim_gates["pairwise_order_accuracy_floor"])
    falsification["claim_gate_passed"] = pd.NA
    fpr_mask = falsification["metric"] == "hard_negative_fpr"
    ordering_mask = falsification["metric"] == "pairwise_order_accuracy"
    falsification.loc[fpr_mask, "claim_gate_passed"] = (
        falsification.loc[fpr_mask, "ci_high"] <= noninferiority_margin
    )
    falsification.loc[ordering_mask, "claim_gate_passed"] = (
        falsification.loc[ordering_mask, "metric_a_ci_low"] > ordering_floor
    )
    falsification["claim_gate_rule"] = pd.NA
    falsification.loc[fpr_mask, "claim_gate_rule"] = (
        f"ci_high <= {noninferiority_margin}"
    )
    falsification.loc[ordering_mask, "claim_gate_rule"] = (
        f"metric_a_ci_low > {ordering_floor}"
    )
    transfer = pd.read_csv(results_dir / "task_cross_task_transfer.csv")
    same_task = pd.read_csv(results_dir / "task_same_task_calibration.csv")
    same_task = same_task.assign(evaluation_regime="same_task")
    transfer = transfer.assign(evaluation_regime="cross_task")
    selected_systems = pd.concat([same_task, transfer], ignore_index=True)
    holdout_low_field = "reference_holdout_alert_rate_ci_low"
    holdout_high_field = "reference_holdout_alert_rate_ci_high"
    if any(
        field not in selected_systems or selected_systems[field].isna().any()
        for field in (holdout_low_field, holdout_high_field)
    ):
        raise ValueError(
            "Frozen selected-system reports lack held-out reference intervals"
        )
    selected_systems["reference_alert_budget_status"] = _reference_budget_status(
        selected_systems[holdout_low_field],
        selected_systems[holdout_high_field],
        budget=reference_budget,
    )
    selected_systems["reference_alert_budget_supported"] = (
        selected_systems["reference_alert_budget_status"] == "supported"
    )
    early_inference_path = results_dir / "early_warning_primary_inference.csv"
    if not early_inference_path.exists():
        raise FileNotFoundError(
            "Claim tables require early_warning_primary_inference.csv"
        )
    early_inference = pd.read_csv(early_inference_path)
    required_early_fields = {
        "endpoint",
        "mean_diff",
        "ci_low",
        "ci_high",
        "p_value",
        "n_cells",
        "status",
    }
    if missing_early := required_early_fields.difference(
        early_inference.columns
    ):
        raise ValueError(
            "Primary early-warning inference lacks fields "
            f"{sorted(missing_early)}"
        )
    if len(early_inference) != 1:
        raise ValueError(
            "Primary early-warning inference must contain exactly one endpoint"
        )
    early_inference["claim_supported"] = (
        (early_inference["ci_low"] > 0.0)
        & (early_inference["p_value"] < 0.05)
    )
    early_inference["claim_gate_rule"] = (
        "hierarchical AUEW uplift ci_low > 0 and p_value < 0.05"
    )
    claims = pd.read_csv(results_dir / "claim_tests.csv") if (results_dir / "claim_tests.csv").exists() else pd.DataFrame()
    robust = pd.read_csv(results_dir / "robustness_summary.csv") if (results_dir / "robustness_summary.csv").exists() else pd.DataFrame()
    controls = pd.read_csv(results_dir / "negative_control_report.csv") if (results_dir / "negative_control_report.csv").exists() and (results_dir / "negative_control_report.csv").stat().st_size > 0 else pd.DataFrame()

    main_table = early_inference.copy()

    supporting_parts = [
        selected_systems.assign(section="static_selected_systems")
    ]
    if not claims.empty:
        supporting_parts.append(claims.assign(section="claim_tests"))
    if not robust.empty:
        supporting_parts.append(robust.assign(section="robustness_summary"))
    supporting_parts.append(
        falsification.assign(section="falsification_significance")
    )
    if not controls.empty:
        supporting_parts.append(controls.assign(section="negative_controls"))
    supporting_table = (
        pd.concat(supporting_parts, ignore_index=True, sort=False)
        if supporting_parts
        else pd.DataFrame()
    )

    main_csv = results_dir / "claim_main_table.csv"
    supporting_csv = results_dir / "claim_supporting_table.csv"
    reference_gate_rows = selected_systems[
        [
            column
            for column in (
                "model",
                "source_task",
                "target_task",
                "probe",
                "k",
                holdout_low_field,
                holdout_high_field,
                "reference_alert_budget_status",
                "reference_alert_budget_supported",
            )
            if column in selected_systems
        ]
    ].copy()
    reference_gate_rows["gate_name"] = "selected_system_reference_alert_budget"
    reference_gate_rows["registered_limit"] = reference_budget
    reference_gate_rows["claim_gate_passed"] = reference_gate_rows[
        "reference_alert_budget_supported"
    ]
    hard_gate_rows = falsification[fpr_mask | ordering_mask][
        [
            "comparison_id",
            "task_name",
            "metric",
            "claim_gate_rule",
            "claim_gate_passed",
        ]
    ].copy()
    hard_gate_rows["gate_name"] = "hard_negative"
    early_gate_rows = early_inference[
        [
            "endpoint",
            "mean_diff",
            "ci_low",
            "ci_high",
            "p_value",
            "claim_gate_rule",
            "claim_supported",
        ]
    ].copy()
    early_gate_rows["gate_name"] = "primary_early_warning_auew"
    early_gate_rows["claim_gate_passed"] = early_gate_rows[
        "claim_supported"
    ]
    gate_status = pd.concat(
        [early_gate_rows, reference_gate_rows, hard_gate_rows],
        ignore_index=True,
        sort=False,
    )
    gate_status.to_csv(results_dir / "claim_gate_status.csv", index=False)
    main_table.to_csv(main_csv, index=False)
    supporting_table.to_csv(supporting_csv, index=False)
    _to_tex(main_table, results_dir / "claim_main_table.tex")
    if not supporting_table.empty:
        _to_tex(supporting_table, results_dir / "claim_supporting_table.tex")

    print(f"saved {main_csv}")
    print(f"saved {supporting_csv}")


if __name__ == "__main__":
    main()
