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
    parser.add_argument(
        "--device",
        default=None,
        help="Optional execution device for model-backed baselines (auto|cpu|cuda|cuda:N|mps).",
    )
    args = parser.parse_args()

    if not args.skip_validate:
        _run([sys.executable, "-m", "cli.validate_multimodel_config", "--config", args.config, "--check_paths"])
    protocol_cmd = [
        sys.executable,
        "-m",
        "cli.run_protocol_multimodel_benchmark",
        "--config",
        args.config,
    ]
    if args.device is not None:
        protocol_cmd.extend(["--device", args.device])
    _run(protocol_cmd)

    results_dir = args.results_dir
    if results_dir is None:
        cfg = yaml.safe_load(Path(args.config).read_text())
        results_dir = str(cfg["results_dir"])
    _run(
        [
            sys.executable,
            "-m",
            "cli.package_paper_artifacts",
            "--results_dir",
            results_dir,
        ]
    )


if __name__ == "__main__":
    main()
