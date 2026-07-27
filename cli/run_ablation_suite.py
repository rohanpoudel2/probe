from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from cli.common import load_yaml, run_cmd


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    out.update(override)
    return out


def _run(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    run_cmd(cmd)


def _validate_run_config(run_cfg: dict, *, ablation_name: str) -> None:
    required = {
        "results_dir",
        "views",
        "layers",
        "probes",
        "k_values",
        "seeds",
        "balance_modes",
        "max_reference_alert_rate",
        "min_reference_groups",
        "models",
    }
    missing = sorted(required.difference(run_cfg))
    if missing:
        raise ValueError(f"Ablation {ablation_name} lacks fields {missing}")
    pairs = [
        row
        for key in ("task_pairs", "calibration_pairs", "transfer_pairs")
        for row in run_cfg.get(key, [])
    ]
    if not pairs:
        raise ValueError(f"Ablation {ablation_name} contains no task pairs")
    required_tasks = {
        str(row[key])
        for row in pairs
        for key in ("source_task", "target_task")
    }
    for model in run_cfg["models"]:
        model_name = str(model.get("name", "unknown"))
        feature_dirs = model.get("feature_dirs") or {}
        missing_tasks = sorted(required_tasks.difference(feature_dirs))
        if missing_tasks:
            raise ValueError(
                f"Ablation {ablation_name} model {model_name} lacks feature "
                f"directories for {missing_tasks}"
            )
        if not model.get("reference_feature_dir"):
            raise ValueError(
                f"Ablation {ablation_name} model {model_name} lacks "
                "reference_feature_dir"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run named ablation suite from YAML config")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Forwarded to downstream benchmark runs (auto|cpu|cuda|cuda:N|mps).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional path for a copy of the aggregate ablation summary.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base = cfg["base"]
    for ablation in cfg["ablations"]:
        name = ablation["name"]
        run_cfg = _merge(base, ablation.get("overrides", {}))
        _validate_run_config(run_cfg, ablation_name=name)
        results_dir = Path(run_cfg["results_dir"]) / name
        tmp_cfg_path = results_dir.parent / f"{name}.generated.yaml"
        run_cfg["results_dir"] = str(results_dir)
        if not args.dry_run:
            tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_cfg_path.write_text(
                yaml.safe_dump(run_cfg, sort_keys=False),
                encoding="utf-8",
            )
        protocol_cmd = [
            sys.executable,
            "-m",
            "cli.run_protocol_multimodel_benchmark",
            "--config",
            str(tmp_cfg_path),
        ]
        if args.device is not None:
            protocol_cmd.extend(["--device", args.device])
        _run(protocol_cmd, dry_run=args.dry_run)

    summary_cmd = [
        sys.executable,
        "-m",
        "cli.summarize_ablation_results",
        "--root_dir",
        str(Path(base["results_dir"])),
    ]
    if args.summary_output:
        summary_cmd.extend(["--output", args.summary_output])
    _run(summary_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
