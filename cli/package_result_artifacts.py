from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _latex_from_csv(csv_path: Path, tex_path: Path, nrows: int | None = None) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return
    if nrows is not None:
        df = df.head(nrows)
    tex_path.write_text(
        df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}"),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package validated result tables, figures, and provenance"
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_root = (
        Path(args.output_dir) if args.output_dir else results_dir / "result_artifacts"
    )
    tables = out_root / "tables"
    figures = out_root / "figures"
    notes = out_root / "notes"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)

    csvs = [
        "claim_main_table.csv",
        "claim_supporting_table.csv",
        "early_warning_source_selection.csv",
        "early_warning_report.csv",
        "early_warning_auew.csv",
        "early_warning_primary_details.csv",
        "early_warning_cell_inference.csv",
        "early_warning_primary_inference.csv",
        "task_primary_source_systems.csv",
        "task_primary_transfer_report.csv",
        "task_cross_model_table.csv",
        "task_cross_task_transfer.csv",
        "task_same_task_calibration.csv",
        "task_fsei.csv",
        "task_significance.csv",
        "falsification_slices.csv",
        "falsification_significance.csv",
        "task_geometry_summary.csv",
        "task_direction_alignment.csv",
        "negative_control_report.csv",
        "sanity_checks.csv",
        "claim_gate_status.csv",
        "ablation_summary.csv",
    ]
    for name in csvs:
        _copy_if_exists(results_dir / name, tables / name)
        _latex_from_csv(
            results_dir / name, tables / name.replace(".csv", ".tex"), nrows=30
        )

    provenance = out_root / "provenance"
    _copy_if_exists(
        results_dir / "falsification_shift_predictions.jsonl",
        provenance / "falsification_shift_predictions.jsonl",
    )
    _copy_if_exists(
        results_dir / "falsification_pair_predictions.jsonl",
        provenance / "falsification_pair_predictions.jsonl",
    )
    falsification_manifests = results_dir / "falsification_manifests"
    if falsification_manifests.exists():
        shutil.copytree(
            falsification_manifests,
            provenance / "falsification_manifests",
            dirs_exist_ok=True,
        )
    protocol_artifacts = results_dir / "protocol_artifacts"
    if protocol_artifacts.exists():
        shutil.copytree(
            protocol_artifacts,
            provenance / "protocol_artifacts",
            dirs_exist_ok=True,
        )

    for name in [
        "cross_model_transfer.png",
        "label_budget_scaling.png",
        "task_transfer_matrix.png",
    ]:
        _copy_if_exists(results_dir / name, figures / name)

    manifest_lines = [
        "Validated result artifact bundle",
        "",
        "Primary result tables:",
        "1. claim_main_table.tex",
        "2. task_cross_task_transfer.tex",
        "3. task_significance.tex",
        "4. falsification_significance.tex",
        "5. claim_gate_status.tex",
        "",
        "Supporting result tables:",
        "1. claim_supporting_table.tex",
        "2. task_cross_model_table.tex",
        "3. task_geometry_summary.tex",
        "4. negative_control_report.tex",
        "5. ablation_summary.tex",
        "6. falsification evidence and immutable protocol artifacts (provenance/)",
        "",
        "Result figures:",
        "1. cross_model_transfer.png",
        "2. label_budget_scaling.png",
        "3. task_transfer_matrix.png",
    ]
    (notes / "result_artifact_manifest.md").write_text(
        "\n".join(manifest_lines),
        encoding="utf-8",
    )
    print(f"packaged result artifacts under {out_root}")


if __name__ == "__main__":
    main()
