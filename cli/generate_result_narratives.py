from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render factual summaries of pre-registered comparisons without result cherry-picking"
    )
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    significance_path = results_dir / "task_significance.csv"
    claims_path = results_dir / "claim_tests.csv"
    if not significance_path.exists() or not claims_path.exists():
        raise FileNotFoundError(
            "Narratives require task_significance.csv and claim_tests.csv from pre-registered inference"
        )
    significance = pd.read_csv(significance_path)
    claims = pd.read_csv(claims_path)
    if significance.empty:
        raise ValueError("No registered comparisons are available for narrative generation")

    main_lines = [
        "Registered result summary",
        "",
        "All statements below correspond to pre-registered paired comparisons; no system was chosen from final-test performance.",
        "",
    ]
    for row in significance.itertuples(index=False):
        main_lines.append(
            f"- {row.comparison_id}: {row.description} Difference={row.mean_diff:.3f}, "
            f"95% hierarchical CI [{row.ci_low:.3f}, {row.ci_high:.3f}], "
            f"Holm-adjusted p={row.holm_adjusted_p_value:.4g}; "
            f"{int(row.n_groups)} scenario groups and {int(row.n_seeds)} training seeds."
        )

    appendix_lines = [
        "Registered claim decisions",
        "",
    ]
    for row in claims.itertuples(index=False):
        appendix_lines.append(
            f"- {row.claim_id}: {'passed' if bool(row.passed) else 'did not pass'} under: {row.decision_rule}."
        )

    controls_path = results_dir / "negative_control_report.csv"
    if controls_path.exists() and controls_path.stat().st_size:
        controls = pd.read_csv(controls_path)
        appendix_lines.extend(
            [
                "",
                f"Negative-control report contains {len(controls)} registered rows. Control gaps are descriptive unless paired scenario-level intervals are included.",
            ]
        )

    (results_dir / "main_text_narrative.md").write_text(
        "\n".join(main_lines) + "\n", encoding="utf-8"
    )
    (results_dir / "appendix_narrative.md").write_text(
        "\n".join(appendix_lines) + "\n", encoding="utf-8"
    )
    print(f"saved {results_dir / 'main_text_narrative.md'}")
    print(f"saved {results_dir / 'appendix_narrative.md'}")


if __name__ == "__main__":
    main()
