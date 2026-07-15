from __future__ import annotations

import argparse
import sys

from cli.common import load_yaml, run_cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command release pipeline for the paper bundle"
    )
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--controls_config", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Forwarded to LLM-judge and other judge-backed protocol runs "
             "(auto|cpu|cuda|cuda:N|mps).",
    )
    parser.add_argument(
        "--run_release_artifacts",
        action="store_true",
        help="Also run sanity checks, claim tests, and final packaging/reporting.",
    )
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    is_frozen = str(base_cfg.get("protocol_stage", "pilot")).lower() == "frozen"

    run_cmd(
        [
            sys.executable,
            "-m",
            "cli.validate_multimodel_config",
            "--config",
            args.base_config,
            "--check_paths",
        ] + (["--final_protocol"] if is_frozen else [])
    )
    protocol_cmd = [
        sys.executable,
        "-m",
        "cli.run_protocol_multimodel_benchmark",
        "--config",
        args.base_config,
    ]
    if args.device is not None:
        protocol_cmd.extend(["--device", args.device])
    run_cmd(protocol_cmd)
    run_cmd(
        [
            sys.executable,
            "-m",
            "cli.run_negative_control_suite",
            "--base_config",
            args.base_config,
            "--controls_config",
            args.controls_config,
            *(
                ["--device", args.device]
                if args.device is not None
                else []
            ),
        ]
    )

    if not is_frozen and not args.run_release_artifacts:
        print(
            "Skipped release-artifact steps because this is a pilot-stage config. "
            "Pass --run_release_artifacts to force final report generation."
        )
        return

    results_dir = str(base_cfg["results_dir"])

    if args.run_release_artifacts or is_frozen:
        run_cmd(
            [
                sys.executable,
                "-m",
                "cli.run_sanity_checks",
                "--results_dir",
                results_dir,
                "--controls_report",
                f"{results_dir}/negative_control_report.csv",
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "cli.run_claim_tests",
                "--results_dir",
                results_dir,
                "--controls_report",
                f"{results_dir}/negative_control_report.csv",
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "cli.build_robustness_summary",
                "--results_dir",
                results_dir,
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "cli.build_final_claim_tables",
                "--results_dir",
                results_dir,
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "cli.package_final_draft_assets",
                "--results_dir",
                results_dir,
            ]
        )


if __name__ == "__main__":
    main()
