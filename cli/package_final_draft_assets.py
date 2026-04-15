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
    tex_path.write_text(df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Package final draft assets for submission")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_root = Path(args.output_dir) if args.output_dir else results_dir / "final_draft_assets"
    tables = out_root / "tables"
    figures = out_root / "figures"
    notes = out_root / "notes"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)

    csvs = [
        "task_cross_model_table.csv",
        "task_cross_task_transfer.csv",
        "task_same_task_calibration.csv",
        "task_fsei.csv",
        "task_significance.csv",
        "task_geometry_summary.csv",
        "task_direction_alignment.csv",
        "task_steering_best.csv",
        "negative_control_report.csv",
        "sanity_checks.csv",
        "ablation_summary.csv",
    ]
    for name in csvs:
        _copy_if_exists(results_dir / name, tables / name)
        _latex_from_csv(results_dir / name, tables / name.replace(".csv", ".tex"), nrows=30)

    for name in ["camera_ready_cross_model.png", "camera_ready_k_sweep.png", "task_transfer_matrix.png", "task_steering_tradeoff.png"]:
        _copy_if_exists(results_dir / name, figures / name)

    manifest_lines = [
        "Final draft asset bundle",
        "",
        "Suggested main paper tables:",
        "1. task_cross_model_table.tex",
        "2. task_cross_task_transfer.tex",
        "3. sanity_checks.tex",
        "",
        "Suggested appendix tables:",
        "1. task_significance.tex",
        "2. task_geometry_summary.tex",
        "3. task_steering_best.tex",
        "4. negative_control_report.tex",
        "5. ablation_summary.tex",
        "",
        "Suggested figures:",
        "1. camera_ready_cross_model.png",
        "2. camera_ready_k_sweep.png",
        "3. task_transfer_matrix.png",
        "4. task_steering_tradeoff.png",
    ]
    (notes / "draft_asset_manifest.md").write_text("\n".join(manifest_lines))
    print(f"packaged final draft assets under {out_root}")


if __name__ == "__main__":
    main()
