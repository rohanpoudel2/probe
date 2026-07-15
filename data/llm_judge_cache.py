from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from data.llm_judge import LLM_JUDGE_CACHE_SCHEMA_VERSION
from data.rollout_schema import canonical_json


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_payload_sha256(split: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(split):
        array = np.asarray(split[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        if array.dtype.kind in {"U", "S", "O"}:
            for value in array.astype(str).reshape(-1):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        else:
            canonical = np.ascontiguousarray(array)
            digest.update(canonical.dtype.str.encode("ascii"))
            digest.update(canonical.tobytes())
    return digest.hexdigest()


def atomic_save_judge_cache(
    path: Path,
    scored_splits: dict[str, dict[str, np.ndarray]],
    metadata: dict[str, Any],
) -> None:
    if not scored_splits:
        raise ValueError("Cannot save an empty judge score cache")
    metadata = dict(metadata)
    metadata["split_payload_sha256"] = {
        split_name: _split_payload_sha256(split)
        for split_name, split in scored_splits.items()
    }
    payload: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(canonical_json(metadata))
    }
    for split_name, split in scored_splits.items():
        for key in (
            "labels",
            "scores",
            "example_ids",
            "question_ids",
            "prompt_sha256",
            "prompt_token_lengths",
        ):
            if key not in split:
                raise ValueError(f"Judge split {split_name} is missing {key}")
            payload[f"{split_name}__{key}"] = np.asarray(split[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def load_judge_cache(
    path: Path,
    *,
    expected_context_hash: str | None = None,
    expected_splits: set[str] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing LLM-judge score cache: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        if "metadata_json" not in bundle:
            raise ValueError(f"Judge cache {path} lacks metadata_json")
        metadata = json.loads(str(bundle["metadata_json"].item()))
        split_names = sorted(
            {
                key.split("__", 1)[0]
                for key in bundle.files
                if "__" in key and key != "metadata_json"
            }
        )
        scored: dict[str, dict[str, np.ndarray]] = {}
        for split_name in split_names:
            required = {
                key: f"{split_name}__{key}"
                for key in (
                    "labels",
                    "scores",
                    "example_ids",
                    "question_ids",
                    "prompt_sha256",
                    "prompt_token_lengths",
                )
            }
            missing = [
                key for key, full_key in required.items() if full_key not in bundle
            ]
            if missing:
                raise ValueError(
                    f"Judge cache {path} split {split_name} lacks {missing}"
                )
            scored[split_name] = {
                "labels": np.asarray(bundle[required["labels"]], dtype=np.int64),
                "scores": np.asarray(bundle[required["scores"]], dtype=float),
                "example_ids": np.asarray(bundle[required["example_ids"]]).astype(str),
                "question_ids": np.asarray(bundle[required["question_ids"]]).astype(
                    str
                ),
                "prompt_sha256": np.asarray(bundle[required["prompt_sha256"]]).astype(
                    str
                ),
                "prompt_token_lengths": np.asarray(
                    bundle[required["prompt_token_lengths"]], dtype=np.int64
                ),
            }

    if metadata.get("schema_version") != LLM_JUDGE_CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported judge cache schema in {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("context_hash", ""))):
        raise ValueError(f"Judge cache {path} has an invalid context hash")
    if expected_context_hash and metadata.get("context_hash") != expected_context_hash:
        raise ValueError(
            f"Judge cache {path} does not match the requested scoring context"
        )
    if expected_splits is not None and set(scored) != set(expected_splits):
        raise ValueError(
            f"Judge cache {path} has splits {sorted(scored)}; expected {sorted(expected_splits)}"
        )
    payload_hashes = metadata.get("split_payload_sha256")
    if not isinstance(payload_hashes, dict) or set(payload_hashes) != set(scored):
        raise ValueError(f"Judge cache {path} lacks complete split payload hashes")
    for split_name, split in scored.items():
        length = len(split["labels"])
        if not length or any(
            array.ndim != 1 or len(array) != length for array in split.values()
        ):
            raise ValueError(f"Judge cache {path} has misaligned split {split_name}")
        if not set(np.unique(split["labels"]).tolist()).issubset({0, 1}):
            raise ValueError(f"Judge cache {path} has non-binary labels")
        if not np.all(np.isfinite(split["scores"])) or np.any(
            (split["scores"] < 0.0) | (split["scores"] > 1.0)
        ):
            raise ValueError(f"Judge cache {path} has invalid probabilities")
        if len(set(split["example_ids"].tolist())) != length:
            raise ValueError(f"Judge cache {path} has duplicate example IDs")
        if any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in split["prompt_sha256"]
        ):
            raise ValueError(f"Judge cache {path} has invalid prompt hashes")
        if np.any(split["prompt_token_lengths"] < 1):
            raise ValueError(f"Judge cache {path} has invalid prompt token lengths")
        if payload_hashes[split_name] != _split_payload_sha256(split):
            raise ValueError(
                f"Judge cache {path} split {split_name} failed its payload hash"
            )
    metadata["cache_file"] = str(path)
    metadata["cache_sha256"] = _file_sha256(path)
    return scored, metadata
