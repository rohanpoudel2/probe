from __future__ import annotations

# This executable script adds the repository root before importing project modules.
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks import TASK_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 scope lock and task audit")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    task_cfg = cfg.get("task_families", {})
    primary = task_cfg.get("primary")
    transfer = task_cfg.get("transfer", [])
    stress = task_cfg.get("stress_tests", [])
    auxiliary = task_cfg.get("auxiliary_controls", [])

    enabled = [primary] + list(transfer) + list(stress) + list(auxiliary)
    enabled = [x for x in enabled if x]

    print("=== Frontier protocol scope lock ===")
    print(f"paper_track: {cfg.get('project', {}).get('target_track', 'unknown')}")
    print(f"primary task: {primary}")
    print(f"transfer tasks: {transfer}")
    print(f"stress tests: {stress}")
    print(f"auxiliary controls: {auxiliary}")
    print()

    print("=== Task specs ===")
    for name in enabled:
        cls = TASK_REGISTRY.get(name)
        if cls is None:
            print(f"[missing] {name}")
            continue
        task = cls()
        print(json.dumps(task.spec.__dict__, indent=2))
        print()

    print("=== Key protocol decisions ===")
    print(
        "1. The primary estimand is white-box gain over the strongest registered transcript monitor."
    )
    print(
        "2. All behavior data must be generated on-policy and labeled after generation."
    )
    print("3. Thresholds are frozen on dedicated benign calibration traffic at 1% FPR.")
    print("4. Scenario groups, not rows or rollouts, are the independent units.")
    print(
        "5. Steering is deferred until monitoring survives leakage and black-box baselines."
    )


if __name__ == "__main__":
    main()
