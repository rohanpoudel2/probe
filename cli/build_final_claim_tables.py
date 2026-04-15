from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _to_tex(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final main-text and appendix claim tables")
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    transfer = pd.read_csv(results_dir / "task_cross_task_transfer.csv")
    claims = pd.read_csv(results_dir / "claim_tests.csv") if (results_dir / "claim_tests.csv").exists() else pd.DataFrame()
    robust = pd.read_csv(results_dir / "robustness_summary.csv") if (results_dir / "robustness_summary.csv").exists() else pd.DataFrame()
    controls = pd.read_csv(results_dir / "negative_control_report.csv") if (results_dir / "negative_control_report.csv").exists() and (results_dir / "negative_control_report.csv").stat().st_size > 0 else pd.DataFrame()

    main_table = transfer[[c for c in ["model", "source_task", "target_task", "transfer_recall_at_1pct_fpr_mean", "transfer_auroc_mean"] if c in transfer.columns]].copy()
    main_table = main_table.sort_values(["model", "source_task", "target_task"]).reset_index(drop=True)

    appendix_parts = []
    if not claims.empty:
        appendix_parts.append(claims.assign(section="claim_tests"))
    if not robust.empty:
        appendix_parts.append(robust.assign(section="robustness_summary"))
    if not controls.empty:
        appendix_parts.append(controls.assign(section="negative_controls"))
    appendix_table = pd.concat(appendix_parts, ignore_index=True, sort=False) if appendix_parts else pd.DataFrame()

    main_csv = results_dir / "claim_main_table.csv"
    appendix_csv = results_dir / "claim_appendix_table.csv"
    main_table.to_csv(main_csv, index=False)
    appendix_table.to_csv(appendix_csv, index=False)
    _to_tex(main_table, results_dir / "claim_main_table.tex")
    if not appendix_table.empty:
        _to_tex(appendix_table, results_dir / "claim_appendix_table.tex")

    print(f"saved {main_csv}")
    print(f"saved {appendix_csv}")


if __name__ == "__main__":
    main()
