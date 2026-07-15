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


def resolve_torch_device(requested: str = "auto") -> str:
    import torch

    device = str(requested or "auto").strip().lower()
    if not device:
        device = "auto"
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if (
            hasattr(torch.backends, "mps")
            and hasattr(torch.backends.mps, "is_available")
            and torch.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"
    if device == "cpu":
        return "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda is not available")
        if device == "cuda":
            return "cuda"
        # Preserve explicit CUDA device requests such as cuda:0.
        requested_index = device.split(":", 1)[1] if ":" in device else None
        if requested_index is None or requested_index == "":
            return "cuda"
        if not requested_index.isdigit():
            raise ValueError(f"Invalid CUDA device string: {requested!r}")
        index = int(requested_index)
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested CUDA device {requested!r} is out of range"
            )
        return f"cuda:{index}"
    if device == "mps":
        if (
            not hasattr(torch.backends, "mps")
            or not hasattr(torch.backends.mps, "is_available")
            or not torch.backends.mps.is_available()
        ):
            raise ValueError("MPS requested but torch.backends.mps is not available")
        return "mps"
    raise ValueError(f"Unsupported device {requested!r}; expected auto|cpu|mps|cuda|cuda:N")
