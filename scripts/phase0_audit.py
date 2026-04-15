from __future__ import annotations

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

    print("=== Phase 0 scope lock ===")
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

    print("=== Key Phase 0 decisions ===")
    print("1. Main paper direction centers sycophancy, transfers to motivated reasoning, and stresses on CoT distortion.")
    print("2. MASK-style honesty analysis is treated as an auxiliary control, not part of the main sycophancy benchmark.")
    print("3. Span-aware features and steering interfaces are first-class parts of the benchmark.")
    print("4. Grouped splitting by question_id is the default assumption for task families.")


if __name__ == "__main__":
    main()
