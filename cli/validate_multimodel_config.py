from __future__ import annotations

import argparse
from pathlib import Path

from cli.common import load_yaml


REQUIRED_TOP_KEYS = ["results_dir", "models"]
REQUIRED_MODEL_KEYS = ["name", "feature_dirs"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multi-model benchmark config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--check_paths", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    for key in REQUIRED_TOP_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing top-level key: {key}")
    if not cfg.get("task_pairs") and not cfg.get("calibration_pairs") and not cfg.get("transfer_pairs"):
        raise ValueError("Config must define task_pairs or calibration_pairs / transfer_pairs")

    for model_cfg in cfg["models"]:
        for key in REQUIRED_MODEL_KEYS:
            if key not in model_cfg:
                raise ValueError(f"Model config missing key: {key}")
        if args.check_paths:
            for task_name, path_str in model_cfg["feature_dirs"].items():
                path = Path(path_str)
                if not path.exists():
                    raise FileNotFoundError(f"Missing feature directory for {model_cfg['name']} / {task_name}: {path}")

    print("config validation passed")


if __name__ == "__main__":
    main()
