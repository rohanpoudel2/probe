from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required camera-ready artifact is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required camera-ready artifact is missing: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package a camera-ready bundle with numbered assets and narratives"
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else results_dir / "camera_ready_bundle"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.build_numbered_draft_assets",
            "--results_dir",
            str(results_dir),
            "--mapping",
            args.mapping,
            "--output_dir",
            str(out_root / "assets"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.generate_result_narratives",
            "--results_dir",
            str(results_dir),
        ],
        check=True,
    )

    _copy(
        results_dir / "main_text_narrative.md",
        out_root / "notes" / "main_text_narrative.md",
    )
    _copy(
        results_dir / "appendix_narrative.md",
        out_root / "notes" / "appendix_narrative.md",
    )
    _copy(
        results_dir / "claim_main_table.tex",
        out_root / "tables" / "claim_main_table.tex",
    )
    _copy(
        results_dir / "claim_appendix_table.tex",
        out_root / "tables" / "claim_appendix_table.tex",
    )
    _copy(results_dir / "claim_tests.csv", out_root / "tables" / "claim_tests.csv")
    _copy(
        results_dir / "robustness_summary.csv",
        out_root / "tables" / "robustness_summary.csv",
    )
    _copy(
        results_dir / "falsification_slices.csv",
        out_root / "tables" / "falsification_slices.csv",
    )
    _copy(
        results_dir / "falsification_significance.csv",
        out_root / "tables" / "falsification_significance.csv",
    )
    _copy(
        results_dir / "falsification_shift_predictions.jsonl",
        out_root / "provenance" / "falsification_shift_predictions.jsonl",
    )
    _copy(
        results_dir / "falsification_pair_predictions.jsonl",
        out_root / "provenance" / "falsification_pair_predictions.jsonl",
    )
    _copy_tree(
        results_dir / "falsification_manifests",
        out_root / "provenance" / "falsification_manifests",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.build_submission_manifest",
            "--bundle_dir",
            str(out_root),
        ],
        check=True,
    )
    print(f"packaged camera-ready bundle under {out_root}")


if __name__ == "__main__":
    main()
