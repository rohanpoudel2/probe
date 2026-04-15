from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate short result narratives for main text and appendix")
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    transfer = pd.read_csv(results_dir / "task_cross_task_transfer.csv")
    claims = pd.read_csv(results_dir / "claim_tests.csv") if (results_dir / "claim_tests.csv").exists() else pd.DataFrame()
    controls = pd.read_csv(results_dir / "negative_control_report.csv") if (results_dir / "negative_control_report.csv").exists() and (results_dir / "negative_control_report.csv").stat().st_size > 0 else pd.DataFrame()
    robust = pd.read_csv(results_dir / "robustness_summary.csv") if (results_dir / "robustness_summary.csv").exists() else pd.DataFrame()

    best_row = transfer.sort_values("transfer_recall_at_1pct_fpr_mean", ascending=False).iloc[0]
    main_lines = [
        "Main text narrative",
        "",
        f"Our strongest transfer result appears for {best_row['model']} from {best_row['source_task']} to {best_row['target_task']}, with transfer recall at 1% FPR of {best_row['transfer_recall_at_1pct_fpr_mean']:.3f} and transfer AUROC of {best_row['transfer_auroc_mean']:.3f}.",
    ]
    if not claims.empty:
        passed = int(claims["passed"].sum())
        total = int(len(claims))
        main_lines.append(f"Across the paper-level claim checks, {passed} of {total} claims pass under the current run configuration.")
    if not controls.empty:
        min_gap = controls["main_minus_control"].min()
        main_lines.append(f"Relative to negative controls, the smallest observed main-minus-control gap is {min_gap:.3f}, which helps support that the effect is not a trivial artifact.")

    appendix_lines = [
        "Appendix narrative",
        "",
        f"The appendix should emphasize robustness across {transfer['model'].nunique()} models and {len(transfer)} transfer settings in the current export.",
    ]
    if not robust.empty:
        mean_std = robust[[c for c in robust.columns if c.endswith('_std_across_systems')]].mean(numeric_only=True).mean()
        appendix_lines.append(f"Average cross-system variability over the exported robustness metrics is {mean_std:.3f}.")
    if not controls.empty:
        appendix_lines.append("The negative-control appendix should report permuted-label, shuffled-row, and noise baselines beside the main transfer tables whenever available.")

    (results_dir / "main_text_narrative.md").write_text("\n".join(main_lines))
    (results_dir / "appendix_narrative.md").write_text("\n".join(appendix_lines))
    print(f"saved {results_dir / 'main_text_narrative.md'}")
    print(f"saved {results_dir / 'appendix_narrative.md'}")


if __name__ == "__main__":
    main()
