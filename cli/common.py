from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml


def run_cmd(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())

