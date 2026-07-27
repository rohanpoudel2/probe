from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from cli.common import (
    bounded_batch_size_for_device,
    inference_dtype_for_device,
    resolve_torch_device,
)


def _set_torch_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool,
    cuda_count: int = 1,
    mps_available: bool = False,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available, raising=False)
    monkeypatch.setattr(
        torch.cuda, "device_count", lambda: cuda_count, raising=False
    )
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: mps_available),
        raising=False,
    )


def test_resolve_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_torch_state(
        monkeypatch, cuda_available=True, cuda_count=2, mps_available=True
    )
    assert resolve_torch_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_torch_state(monkeypatch, cuda_available=False, mps_available=True)
    assert resolve_torch_device("auto") == "mps"


def test_resolve_device_cpu_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_torch_state(monkeypatch, cuda_available=True, cuda_count=4, mps_available=True)
    assert resolve_torch_device("cpu") == "cpu"


def test_resolve_device_rejects_missing_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_torch_state(monkeypatch, cuda_available=False, mps_available=False)
    with pytest.raises(ValueError, match="CUDA requested"):
        resolve_torch_device("cuda")


def test_resolve_device_rejects_missing_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_torch_state(monkeypatch, cuda_available=False, mps_available=False)
    with pytest.raises(ValueError, match="MPS requested"):
        resolve_torch_device("mps")


def test_mps_uses_memory_efficient_supported_inference_dtype() -> None:
    assert inference_dtype_for_device("mps") is torch.float16
    assert inference_dtype_for_device("cpu") == "auto"


def test_mps_caps_padded_transformer_batches() -> None:
    assert bounded_batch_size_for_device(16, "mps") == 1
    assert bounded_batch_size_for_device(16, "cuda") == 16
