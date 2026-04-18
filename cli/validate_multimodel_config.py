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
    parser.add_argument(
        "--allow_missing_models",
        action="store_true",
        help="Skip models whose feature directories do not exist instead of failing. "
        "Requires at least one model to be fully present.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    for key in REQUIRED_TOP_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing top-level key: {key}")
    if not cfg.get("task_pairs") and not cfg.get("calibration_pairs") and not cfg.get("transfer_pairs"):
        raise ValueError("Config must define task_pairs or calibration_pairs / transfer_pairs")

    present_models: list[str] = []
    skipped_models: list[tuple[str, list[str]]] = []

    for model_cfg in cfg["models"]:
        for key in REQUIRED_MODEL_KEYS:
            if key not in model_cfg:
                raise ValueError(f"Model config missing key: {key}")
        if args.check_paths:
            missing = [
                f"{task_name}: {path_str}"
                for task_name, path_str in model_cfg["feature_dirs"].items()
                if not Path(path_str).exists()
            ]
            if missing:
                if args.allow_missing_models:
                    skipped_models.append((model_cfg["name"], missing))
                    continue
                raise FileNotFoundError(
                    f"Missing feature directory for {model_cfg['name']} / {missing[0]}"
                )
        present_models.append(model_cfg["name"])

    if skipped_models:
        for name, missing in skipped_models:
            print(f"skipped {name}: {len(missing)} missing path(s)")
            for item in missing:
                print(f"  - {item}")
        if args.check_paths and not present_models:
            raise FileNotFoundError("No model has all feature directories present")
        print(f"validated {len(present_models)} model(s): {', '.join(present_models)}")

    print("config validation passed")


if __name__ == "__main__":
    main()
