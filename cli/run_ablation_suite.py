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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run named ablation suite from YAML config")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base = cfg["base"]
    for ablation in cfg["ablations"]:
        name = ablation["name"]
        run_cfg = _merge(base, ablation.get("overrides", {}))
        results_dir = Path(run_cfg["results_dir"]) / name
        tmp_cfg_path = results_dir.parent / f"{name}.generated.yaml"
        tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        run_cfg["results_dir"] = str(results_dir)
        tmp_cfg_path.write_text(yaml.safe_dump(run_cfg, sort_keys=False))
        run_cmd([sys.executable, "-m", "cli.run_protocol_multimodel_benchmark", "--config", str(tmp_cfg_path)])

    run_cmd([sys.executable, "-m", "cli.summarize_ablation_results", "--root_dir", str(Path(base["results_dir"]))])


if __name__ == "__main__":
    main()
