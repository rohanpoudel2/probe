from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper benchmark through packaged artifacts")
    parser.add_argument("--config", required=True, help="Protocol benchmark config YAML")
    parser.add_argument("--results_dir", default=None, help="Optional override for artifact packaging input")
    parser.add_argument("--skip_validate", action="store_true")
    args = parser.parse_args()

    if not args.skip_validate:
        _run([sys.executable, "validate_multimodel_config.py", "--config", args.config, "--check_paths"])
    _run([sys.executable, "run_protocol_multimodel_benchmark.py", "--config", args.config])

    results_dir = args.results_dir
    if results_dir is None:
        cfg = yaml.safe_load(Path(args.config).read_text())
        results_dir = str(cfg["results_dir"])
    _run([sys.executable, "package_paper_artifacts.py", "--results_dir", results_dir])


if __name__ == "__main__":
    main()
