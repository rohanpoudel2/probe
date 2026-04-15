from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cli.common import load_yaml, run_cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 multi-model task benchmark from YAML config")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    overwrite_consumed = False
    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        feature_dirs = model_cfg["feature_dirs"]
        for pair in cfg["task_pairs"]:
            source_task = pair["source_task"]
            target_task = pair["target_task"]
            cmd = [
                sys.executable, "-m", "cli.run_task_sweep",
                "--source_dir", feature_dirs[source_task],
                "--source_task", source_task,
                "--target_dir", feature_dirs[target_task],
                "--target_task", target_task,
                "--model", model_name,
                "--results_dir", str(results_dir),
                "--views", cfg.get("views", "full_text,answer"),
                "--layers", str(cfg.get("layers", "all")),
                "--probes", cfg.get("probes", "P1_logistic,P2_mass_mean,P3_lda,P4_cosine,P7_mahalanobis"),
                "--k_values", cfg.get("k_values", "1,2,4,8"),
                "--seeds", str(cfg.get("seeds", 5)),
                "--balance_modes", cfg.get("balance_modes", "balanced,imbalanced"),
            ]
            if cfg.get("overwrite", False) and not overwrite_consumed:
                cmd.append("--overwrite")
                overwrite_consumed = True
            run_cmd(cmd)

    run_cmd([sys.executable, "-m", "cli.aggregate_task_results", "--results_dir", str(results_dir)])
    run_cmd([sys.executable, "-m", "cli.build_frozen_transfer_report", "--results_dir", str(results_dir)])
    run_cmd([sys.executable, "-m", "cli.build_cross_model_tables", "--results_dir", str(results_dir)])
    run_cmd([sys.executable, "-m", "cli.compute_task_significance", "--results_dir", str(results_dir)])
    run_cmd([sys.executable, "-m", "cli.build_transfer_matrix", "--results_dir", str(results_dir)])
    run_cmd([sys.executable, "-m", "cli.plot_camera_ready_task_results", "--results_dir", str(results_dir)])
    run_cmd([sys.executable, "-m", "cli.build_geometry_report", "--config", args.config, "--results_dir", str(results_dir)])
    run_cmd([sys.executable, "-m", "cli.run_steering_suite", "--config", args.config, "--results_dir", str(results_dir)])


if __name__ == "__main__":
    main()
