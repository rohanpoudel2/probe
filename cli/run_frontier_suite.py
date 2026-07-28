from __future__ import annotations

import argparse
from pathlib import Path
import sys

from cli.common import load_yaml, run_cmd


def _is_frozen_stage(cfg: dict) -> bool:
    return str(cfg.get("protocol_stage", "selection")).strip().lower() == "frozen"


def _artifact_inputs(cfg: dict) -> list[tuple[str, Path]]:
    required: list[tuple[str, Path]] = []
    for model_cfg in cfg.get("models", []):
        required.extend(
            (f"{model_cfg['name']} feature {task}", Path(path))
            for task, path in (model_cfg.get("feature_dirs", {}) or {}).items()
        )
        reference_features = model_cfg.get("reference_feature_dir")
        if reference_features:
            required.append(
                (f"{model_cfg['name']} reference features", Path(reference_features))
            )
        if cfg.get("run_black_box_baselines", False):
            for task, path in (model_cfg.get("labeled_data", {}) or {}).items():
                required.append(
                    (f"{model_cfg['name']} labeled data {task}", Path(path))
                )
            reference_data = model_cfg.get("reference_data")
            if reference_data:
                required.append(
                    (f"{model_cfg['name']} reference traffic", Path(reference_data))
                )
            for task, path in (model_cfg.get("text_embedding_cache_dirs", {}) or {}).items():
                required.append(
                    (f"{model_cfg['name']} embedding cache {task}", Path(path))
                )
            reference_cache = model_cfg.get("reference_embedding_cache_dir")
            if reference_cache:
                required.append(
                    (
                        f"{model_cfg['name']} reference embedding cache",
                        Path(reference_cache),
                    )
                )
    for key in ("comparisons_file", "falsification_comparisons_file"):
        value = cfg.get(key)
        if value:
            required.append((f"registered comparisons: {key}", Path(value)))
    falsification_registry = cfg.get("falsification_registry")
    if falsification_registry:
        required.append(("falsification registry", Path(falsification_registry)))
    if cfg.get("run_falsification_suite", False):
        for model_cfg in cfg.get("models", []):
            model_name = str(model_cfg.get("name", "unknown"))
            for task, path in (model_cfg.get("falsification_manifests", {}) or {}).items():
                required.append((f"{model_name} falsification manifest {task}", Path(path)))
    return required


def _print_artifact_status(cfg_path: Path, cfg: dict) -> None:
    print(f"Loaded protocol config: {cfg_path}")
    print(
        f"protocol_version={cfg.get('protocol_version','(unset)')} "
        f"stage={cfg.get('protocol_stage','selection')}"
    )
    model_names = [str(model_cfg.get("name", "unknown")) for model_cfg in cfg.get("models", [])]
    print(f"models={', '.join(model_names) if model_names else '(none)'}")
    print("required run-time artifacts:")

    missing: list[str] = []
    for label, path in _artifact_inputs(cfg):
        status = "PRESENT" if path.exists() else "MISSING"
        print(f"  [{status}] {label}: {path}")
        if status == "MISSING":
            missing.append(f"{label}: {path}")
    if missing:
        print("Missing required artifacts:")
        for line in missing:
            print(f"  - {line}")
    else:
        print(
            "All declared artifacts are present; the validation command still "
            "checks schema, provenance, and completeness."
        )


def _command_preview(cmd: list[str], *, dry_run: bool) -> None:
    print(f"{'DRY-RUN: ' if dry_run else ''}RUN {' '.join(cmd)}")
    if not dry_run:
        run_cmd(cmd)


def _build_commands(
    cfg_path: Path,
    controls_config: Path | None,
    ablation_config: Path | None,
    *,
    include_controls: bool,
    include_ablations: bool,
    include_release_steps: bool,
    judge_device: str | None,
    selection_only: bool = False,
    results_dir_override: str | None = None,
) -> list[tuple[str, list[str]]]:
    cfg = load_yaml(cfg_path)
    effective_results_dir = str(results_dir_override or cfg["results_dir"])
    commands: list[tuple[str, list[str]]] = []
    validate_cmd = [
        sys.executable,
        "-m",
        "cli.validate_multimodel_config",
        "--config",
        str(cfg_path),
        "--check_paths",
    ]
    if _is_frozen_stage(cfg):
        validate_cmd.append("--final_protocol")
    commands.append(("validate_protocol", validate_cmd))

    protocol_cmd = [
        sys.executable,
        "-m",
        "cli.run_protocol_multimodel_benchmark",
        "--config",
        str(cfg_path),
    ]
    if judge_device:
        protocol_cmd.extend(["--device", judge_device])
    if selection_only:
        protocol_cmd.append("--selection-only")
    if results_dir_override:
        protocol_cmd.extend(["--results-dir", results_dir_override])
    commands.append(("run_main_protocol", protocol_cmd))

    if include_controls:
        if controls_config is None:
            raise FileNotFoundError(
                "controls_config was not provided; pass --controls_config or set --no-controls"
            )
        cmd = [
            sys.executable,
            "-m",
            "cli.run_negative_control_suite",
            "--base_config",
            str(cfg_path),
            "--controls_config",
            str(controls_config),
        ]
        if judge_device:
            cmd.extend(["--device", judge_device])
        cmd.extend(["--main_results_dir", effective_results_dir])
        commands.append(("run_negative_control_suite", cmd))

    if include_ablations:
        if ablation_config is None:
            raise FileNotFoundError(
                "ablation_config was not provided; pass --ablation_config or set --no-ablations"
            )
        cmd = [
            sys.executable,
            "-m",
            "cli.run_ablation_suite",
            "--config",
            str(ablation_config),
        ]
        if judge_device:
            cmd.extend(["--device", judge_device])
        if _is_frozen_stage(cfg):
            cmd.extend(
                [
                    "--summary-output",
                    str(Path(effective_results_dir) / "ablation_summary.csv"),
                ]
            )
        commands.append(("run_ablation_suite", cmd))

    if include_release_steps and _is_frozen_stage(cfg):
        results_dir = effective_results_dir
        commands.extend(
            [
                (
                    "run_sanity_checks",
                    [
                        sys.executable,
                        "-m",
                        "cli.run_sanity_checks",
                        "--results_dir",
                        results_dir,
                        "--controls_report",
                        f"{results_dir}/negative_control_report.csv",
                    ],
                ),
                (
                    "run_claim_tests",
                    [
                        sys.executable,
                        "-m",
                        "cli.run_claim_tests",
                        "--results_dir",
                        results_dir,
                        "--controls_report",
                        f"{results_dir}/negative_control_report.csv",
                    ],
                ),
                (
                    "build_robustness_summary",
                    [
                        sys.executable,
                        "-m",
                        "cli.build_robustness_summary",
                        "--results_dir",
                        results_dir,
                    ],
                ),
                (
                    "build_final_claim_tables",
                    [
                        sys.executable,
                        "-m",
                        "cli.build_final_claim_tables",
                        "--results_dir",
                        results_dir,
                        "--config",
                        str(cfg_path),
                    ],
                ),
                (
                    "package_result_artifacts",
                    [
                        sys.executable,
                        "-m",
                        "cli.package_result_artifacts",
                        "--results_dir",
                        results_dir,
                    ],
                ),
            ]
        )

    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete frontier experiment suite from a protocol config."
    )
    parser.add_argument("--config", default="experiments/protocol/main_research_manifest.yaml")
    parser.add_argument(
        "--controls_config",
        default="experiments/controls/negative_controls.yaml",
    )
    parser.add_argument("--ablation_config", default="experiments/protocol/ablation_suite.yaml")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument(
        "--release-artifacts",
        action="store_true",
        help="Run sanity checks, claim tests, robustness summary, and final claim tables. "
             "Only used when protocol_stage is frozen.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--device",
        default=None,
        help="Optional execution device passed to model-backed runs "
             "(auto|cpu|cuda|cuda:N|mps).",
    )
    parser.add_argument(
        "--judge-device",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip --check_paths validation step and proceed directly.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Run source-only system selection and skip controls, ablations, and release steps.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override the protocol results directory.",
    )
    args = parser.parse_args()

    judge_device = (
        args.device if args.device is not None else args.judge_device
    )
    if args.device is not None and args.judge_device is not None:
        print(
            "Both --device and --judge-device were provided; "
            "using --device and ignoring --judge-device."
        )

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Protocol config not found: {cfg_path}")
    cfg = load_yaml(cfg_path)
    selection_only = bool(
        args.selection_only or cfg.get("execution_mode") == "selection"
    )

    controls_path = Path(args.controls_config)
    ablation_path = Path(args.ablation_config)

    if args.no_controls:
        controls_path = None
    elif not controls_path.exists():
        raise FileNotFoundError(f"Controls config not found: {controls_path}")
    if args.no_ablations:
        ablation_path = None
    elif not ablation_path.exists():
        raise FileNotFoundError(f"Ablation config not found: {ablation_path}")

    print("Frontier suite preflight:")
    _print_artifact_status(cfg_path, cfg)

    commands = _build_commands(
        cfg_path=cfg_path,
        controls_config=controls_path,
        ablation_config=ablation_path,
        include_controls=not args.no_controls and not selection_only,
        include_ablations=not args.no_ablations and not selection_only,
        include_release_steps=args.release_artifacts and not selection_only,
        judge_device=judge_device,
        selection_only=selection_only,
        results_dir_override=args.results_dir,
    )
    if args.skip_validation:
        commands = [entry for entry in commands if entry[0] != "validate_protocol"]

    for _, cmd in commands:
        _command_preview(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
