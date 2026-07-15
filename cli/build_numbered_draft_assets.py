from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def _copy(src: Path, dst: Path, *, optional: bool) -> None:
    if not src.exists():
        if optional:
            return
        raise FileNotFoundError(f"Required mapped asset is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy final assets into numbered figure and table names")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    mapping = yaml.safe_load(Path(args.mapping).read_text())
    out_root = Path(args.output_dir) if args.output_dir else results_dir / "numbered_draft_assets"

    for item in mapping.get("tables", []):
        _copy(
            results_dir / item["source"],
            out_root / "tables" / item["target"],
            optional=bool(item.get("optional", False)),
        )
    for item in mapping.get("figures", []):
        _copy(
            results_dir / item["source"],
            out_root / "figures" / item["target"],
            optional=bool(item.get("optional", False)),
        )

    print(f"saved numbered assets under {out_root}")


if __name__ == "__main__":
    main()
