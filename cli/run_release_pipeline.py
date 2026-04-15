from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cli.common import load_yaml, run_cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command release pipeline for the paper bundle")
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--controls_config", required=True)
    args = parser.parse_args()

    run_cmd([sys.executable, "-m", "cli.validate_multimodel_config", "--config", args.base_config, "--check_paths"])
    run_cmd([sys.executable, "-m", "cli.run_protocol_multimodel_benchmark", "--config", args.base_config])
    run_cmd([sys.executable, "-m", "cli.run_negative_control_suite", "--base_config", args.base_config, "--controls_config", args.controls_config])

    base_cfg = load_yaml(args.base_config)
    results_dir = str(base_cfg["results_dir"])

    run_cmd([sys.executable, "-m", "cli.run_sanity_checks", "--results_dir", results_dir, "--controls_report", f"{results_dir}/negative_control_report.csv"])
    run_cmd([sys.executable, "-m", "cli.run_claim_tests", "--results_dir", results_dir, "--controls_report", f"{results_dir}/negative_control_report.csv"])
    run_cmd([sys.executable, "-m", "cli.build_robustness_summary", "--results_dir", results_dir])
    run_cmd([sys.executable, "-m", "cli.build_final_claim_tables", "--results_dir", results_dir])
    run_cmd([sys.executable, "-m", "cli.package_final_draft_assets", "--results_dir", results_dir])


if __name__ == "__main__":
    main()
