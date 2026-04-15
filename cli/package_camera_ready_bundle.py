from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _copy(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a camera-ready bundle with numbered assets and narratives")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_root = Path(args.output_dir) if args.output_dir else results_dir / "camera_ready_bundle"
    out_root.mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, "-m", "cli.build_numbered_draft_assets", "--results_dir", str(results_dir), "--mapping", args.mapping, "--output_dir", str(out_root / "assets")], check=True)
    subprocess.run([sys.executable, "-m", "cli.generate_result_narratives", "--results_dir", str(results_dir)], check=True)

    _copy(results_dir / "main_text_narrative.md", out_root / "notes" / "main_text_narrative.md")
    _copy(results_dir / "appendix_narrative.md", out_root / "notes" / "appendix_narrative.md")
    _copy(results_dir / "claim_main_table.tex", out_root / "tables" / "claim_main_table.tex")
    _copy(results_dir / "claim_appendix_table.tex", out_root / "tables" / "claim_appendix_table.tex")
    _copy(results_dir / "claim_tests.csv", out_root / "tables" / "claim_tests.csv")
    _copy(results_dir / "robustness_summary.csv", out_root / "tables" / "robustness_summary.csv")

    subprocess.run([sys.executable, "-m", "cli.build_submission_manifest", "--bundle_dir", str(out_root)], check=True)
    print(f"packaged camera-ready bundle under {out_root}")


if __name__ == "__main__":
    main()
