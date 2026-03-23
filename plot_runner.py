"""Wrapper for generating benchmark figures from analyzed CSV outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark figures")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir = args.results_dir or config.get("results_dir", "./results")
    subprocess.run(
        [sys.executable, "-m", "evaluation.plots", "--results-dir", str(Path(results_dir))],
        check=True,
    )


if __name__ == "__main__":
    main()
