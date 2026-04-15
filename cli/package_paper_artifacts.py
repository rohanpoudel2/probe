from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _write_latex(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        return
    path.write_text(df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Package main-text and appendix paper artifacts")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_root = Path(args.output_dir) if args.output_dir else results_dir / "paper_artifacts"
    main_dir = out_root / "main"
    appendix_dir = out_root / "appendix"
    main_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir.mkdir(parents=True, exist_ok=True)

    main_cross = _safe_read(results_dir / "task_cross_model_table.csv")
    main_transfer = _safe_read(results_dir / "task_cross_task_transfer.csv")
    signif = _safe_read(results_dir / "task_significance.csv")
    ablation = _safe_read(results_dir / "ablation_summary.csv")
    summary = _safe_read(results_dir / "task_summary.csv")
    fsei = _safe_read(results_dir / "task_fsei.csv")
    rankings = _safe_read(results_dir / "task_cross_model_rankings.csv")
    geometry = _safe_read(results_dir / "task_geometry_summary.csv")
    steering = _safe_read(results_dir / "task_steering_best.csv")

    if main_cross is not None:
        shutil.copy2(results_dir / "task_cross_model_table.csv", main_dir / "task_cross_model_table.csv")
        _write_latex(main_cross.reset_index(), main_dir / "task_cross_model_table.tex")
    if main_transfer is not None:
        condensed = main_transfer[[c for c in ["model", "source_task", "target_task", "transfer_recall_at_1pct_fpr_mean", "transfer_auroc_mean"] if c in main_transfer.columns]]
        condensed.to_csv(main_dir / "task_cross_task_transfer_main.csv", index=False)
        _write_latex(condensed, main_dir / "task_cross_task_transfer_main.tex")
    if signif is not None:
        shutil.copy2(results_dir / "task_significance.csv", appendix_dir / "task_significance.csv")
        _write_latex(signif, appendix_dir / "task_significance.tex")
    if ablation is not None and not ablation.empty:
        ablation.to_csv(appendix_dir / "ablation_summary.csv", index=False)
        _write_latex(ablation, appendix_dir / "ablation_summary.tex")
    if summary is not None:
        summary.head(50).to_csv(appendix_dir / "task_summary_head.csv", index=False)
    if fsei is not None:
        fsei.to_csv(appendix_dir / "task_fsei.csv", index=False)
        _write_latex(fsei, appendix_dir / "task_fsei.tex")
    if rankings is not None:
        rankings.to_csv(appendix_dir / "task_cross_model_rankings.csv", index=False)
        _write_latex(rankings.head(30), appendix_dir / "task_cross_model_rankings.tex")
    if geometry is not None:
        geometry.head(50).to_csv(appendix_dir / "task_geometry_summary.csv", index=False)
        _write_latex(geometry.head(30), appendix_dir / "task_geometry_summary.tex")
    if steering is not None:
        steering.to_csv(appendix_dir / "task_steering_best.csv", index=False)
        _write_latex(steering.head(30), appendix_dir / "task_steering_best.tex")

    for img_name in ["camera_ready_cross_model.png", "camera_ready_k_sweep.png", "task_transfer_matrix.png", "task_steering_tradeoff.png"]:
        src = results_dir / img_name
        if src.exists():
            shutil.copy2(src, main_dir / img_name)

    print(f"packaged paper artifacts under {out_root}")


if __name__ == "__main__":
    main()
