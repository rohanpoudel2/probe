from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from cli.common import load_yaml, run_cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run negative-control benchmark suite")
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--controls_config", required=True)
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    controls_cfg = load_yaml(args.controls_config)

    base_results_root = Path(controls_cfg["control_results_root"])
    control_features_root = Path(controls_cfg["control_features_root"])

    for control in controls_cfg["controls"]:
        control_name = control["name"]
        control_type = control["control_type"]
        run_cfg = load_yaml(args.base_config)
        run_cfg["results_dir"] = str(base_results_root / control_name)
        run_cfg["overwrite"] = True

        for model_cfg in run_cfg["models"]:
            for task_name, src_dir in list(model_cfg["feature_dirs"].items()):
                out_dir = control_features_root / control_name / model_cfg["name"] / task_name
                run_cmd([
                    sys.executable,
                    "-m",
                    "cli.make_control_features",
                    "--input_dir", str(src_dir),
                    "--output_dir", str(out_dir),
                    "--control_type", control_type,
                    "--seed", str(control.get("seed", 0)),
                ])
                model_cfg["feature_dirs"][task_name] = str(out_dir)

        tmp_cfg = control_features_root / f"{control_name}.generated.yaml"
        tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
        tmp_cfg.write_text(yaml.safe_dump(run_cfg, sort_keys=False))
        run_cmd([sys.executable, "-m", "cli.run_protocol_multimodel_benchmark", "--config", str(tmp_cfg)])

    run_cmd([sys.executable, "-m", "cli.build_negative_control_report", "--main_results_dir", str(base_cfg["results_dir"]), "--controls_root", str(base_results_root)])


if __name__ == "__main__":
    main()
