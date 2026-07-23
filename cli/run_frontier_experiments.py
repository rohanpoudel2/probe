from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cli.common import run_cmd


def _add_device(cmd: list[str], device: str | None) -> list[str]:
    if device is None:
        return cmd
    cmd.extend(["--device", device])
    return cmd


def _build_frontier_suite_cmd(
    *,
    config: str,
    controls_config: str | None,
    ablation_config: str | None,
    include_controls: bool,
    include_ablations: bool,
    release_artifacts: bool,
    device: str | None,
    dry_run: bool,
    skip_validation: bool,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "cli.run_frontier_suite",
        "--config",
        str(config),
    ]
    if include_controls and controls_config:
        cmd.extend(["--controls_config", str(controls_config)])
    else:
        cmd.append("--no-controls")
    if not include_ablations or ablation_config is None:
        cmd.append("--no-ablations")
    else:
        cmd.extend(["--ablation_config", str(ablation_config)])
    if release_artifacts:
        cmd.append("--release-artifacts")
    if dry_run:
        cmd.append("--dry-run")
    if skip_validation:
        cmd.append("--skip-validation")
    _add_device(cmd, device)
    return cmd


def _build_ablation_suite_cmd(
    *,
    config: str,
    device: str | None,
    dry_run: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "cli.run_ablation_suite",
        "--config",
        str(config),
    ]
    if dry_run:
        cmd.append("--dry-run")
    _add_device(cmd, device)
    return cmd


def _run(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    run_cmd(cmd)


def _ensure_config_exists(path: str | None, *, label: str) -> None:
    if path is None:
        return
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full frontier research experiment portfolio."
    )
    parser.add_argument(
        "--main-config",
        default="experiments/protocol/main_research_manifest.yaml",
        help="Main protocol manifest.",
    )
    parser.add_argument(
        "--honesty-config",
        default="experiments/controls/honesty_auxiliary_manifest.yaml",
        help="Auxiliary honesty-control manifest.",
    )
    parser.add_argument(
        "--appendix-ablation-config",
        default=None,
        help="Optional appendix ablation suite config.",
    )
    parser.add_argument(
        "--controls-config",
        default="experiments/controls/negative_controls.yaml",
        help="Shared control manifest for frontier protocol runs.",
    )
    parser.add_argument(
        "--main-ablation-config",
        default="experiments/protocol/ablation_suite.yaml",
        help="Main-suite ablation configuration.",
    )
    parser.add_argument("--no-main", action="store_true")
    parser.add_argument("--no-honesty", action="store_true")
    parser.add_argument(
        "--include-honesty-controls",
        action="store_true",
        help="Run negative controls for the honesty auxiliary manifest.",
    )
    parser.add_argument(
        "--run-appendix-ablations",
        action="store_true",
        help="Run appendix-style ablations after the main protocol.",
    )
    parser.add_argument("--device", default=None, help="Execution device for model-backed runs.")
    parser.add_argument("--release-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    # Validate requested configs exist now so the command set fails before start.
    if not args.no_main:
        _ensure_config_exists(args.main_config, label="Main config")
    if not args.no_main or args.include_honesty_controls:
        _ensure_config_exists(args.controls_config, label="Controls config")
    if not args.no_main:
        _ensure_config_exists(args.main_ablation_config, label="Main ablation config")
    if args.appendix_ablation_config is not None:
        _ensure_config_exists(
            args.appendix_ablation_config,
            label="Appendix ablation config",
        )
    if not args.no_honesty:
        _ensure_config_exists(args.honesty_config, label="Honesty auxiliary config")

    if args.no_main and args.no_honesty and not args.run_appendix_ablations:
        raise RuntimeError(
            "No experiments selected. Remove --no-main or --no-honesty, or pass "
            "--run-appendix-ablations."
        )

    commands: list[list[str]] = []
    if not args.no_main:
        commands.append(
            _build_frontier_suite_cmd(
                config=args.main_config,
                controls_config=args.controls_config,
                ablation_config=args.main_ablation_config,
                include_controls=True,
                include_ablations=True,
                release_artifacts=args.release_artifacts,
                device=args.device,
                dry_run=args.dry_run,
                skip_validation=args.skip_validation,
            )
        )

    if not args.no_honesty:
        commands.append(
            _build_frontier_suite_cmd(
                config=args.honesty_config,
                controls_config=args.controls_config if args.include_honesty_controls else None,
                ablation_config=None,
                include_controls=args.include_honesty_controls,
                include_ablations=False,
                release_artifacts=args.release_artifacts,
                device=args.device,
                dry_run=args.dry_run,
                skip_validation=args.skip_validation,
            )
        )

    if args.run_appendix_ablations and args.appendix_ablation_config is not None:
        commands.append(
            _build_ablation_suite_cmd(
                config=args.appendix_ablation_config,
                device=args.device,
                dry_run=args.dry_run,
            )
        )

    for command in commands:
        # Dry-run is handled either in command builders (protocol suites) or here
        # for compatibility with this wrapper.
        _run(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
