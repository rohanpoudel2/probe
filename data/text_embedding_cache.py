from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np


TEXT_EMBEDDING_SCHEMA_VERSION = "text-embedding-cache-v1"
COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VALID_SPLITS = {"train", "calibration", "eval", "test"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_save_text_embedding_cache(
    path: Path,
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {**arrays, **{key: np.asarray(value) for key, value in metadata.items()}}
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _scalar(bundle: Any, key: str) -> Any:
    if key not in bundle:
        raise ValueError(f"Text embedding cache is missing {key}")
    value = bundle[key]
    if np.asarray(value).ndim != 0:
        raise ValueError(f"Text embedding metadata {key} must be scalar")
    return value.item()


def load_text_embedding_cache(
    path: Path, *, require_clean_code: bool = True
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing text embedding cache: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        for key in (
            "embeddings",
            "labels",
            "example_ids",
            "question_ids",
            "protocol_splits",
            "text_sha256",
            "normalized_text_sha256",
            "embedding_input_sha256",
            "original_token_lengths",
            "truncated",
        ):
            if key not in bundle:
                raise ValueError(f"Text embedding cache {path} is missing {key}")
        embeddings = np.asarray(bundle["embeddings"], dtype=np.float32)
        labels = np.asarray(bundle["labels"], dtype=np.int64)
        example_ids = np.asarray(bundle["example_ids"]).astype(str)
        question_ids = np.asarray(bundle["question_ids"]).astype(str)
        protocol_splits = np.asarray(bundle["protocol_splits"]).astype(str)
        text_hashes = np.asarray(bundle["text_sha256"]).astype(str)
        normalized_text_hashes = np.asarray(bundle["normalized_text_sha256"]).astype(str)
        input_hashes = np.asarray(bundle["embedding_input_sha256"]).astype(str)
        token_lengths = np.asarray(bundle["original_token_lengths"], dtype=np.int64)
        truncated = np.asarray(bundle["truncated"], dtype=bool)
        metadata_keys = (
            "schema_version",
            "task_name",
            "view",
            "dataset_sha256",
            "code_revision",
            "code_dirty",
            "embedding_model_id",
            "embedding_model_revision",
            "embedding_tokenizer_revision",
            "embedding_spec_sha256",
            "embedding_config_sha256",
            "pooling",
            "padding_side",
            "normalized",
            "max_length",
            "instruction",
            "instruction_format",
            "monitored_model_id",
            "monitored_model_revision",
            "monitored_tokenizer_revision",
        )
        metadata = {key: _scalar(bundle, key) for key in metadata_keys}

    if metadata["schema_version"] != TEXT_EMBEDDING_SCHEMA_VERSION:
        raise ValueError(f"Unsupported text embedding cache schema in {path}")
    if embeddings.ndim != 2 or not len(embeddings) or not np.all(np.isfinite(embeddings)):
        raise ValueError(f"Text embedding cache {path} has invalid embeddings")
    aligned = (
        labels,
        example_ids,
        question_ids,
        protocol_splits,
        text_hashes,
        normalized_text_hashes,
        input_hashes,
        token_lengths,
        truncated,
    )
    if any(array.ndim != 1 or len(array) != len(embeddings) for array in aligned):
        raise ValueError(f"Text embedding cache {path} has misaligned arrays")
    if len(set(example_ids.tolist())) != len(example_ids):
        raise ValueError(f"Text embedding cache {path} has duplicate example IDs")
    if np.any(np.char.str_len(question_ids) == 0):
        raise ValueError(f"Text embedding cache {path} has empty question IDs")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError(f"Text embedding cache {path} has non-binary labels")
    if not set(np.unique(protocol_splits).tolist()).issubset(VALID_SPLITS):
        raise ValueError(f"Text embedding cache {path} has invalid protocol splits")
    if any(
        not SHA256_RE.fullmatch(value)
        for value in (
            text_hashes.tolist()
            + normalized_text_hashes.tolist()
            + input_hashes.tolist()
        )
    ):
        raise ValueError(f"Text embedding cache {path} has invalid text hashes")
    for key in ("dataset_sha256", "embedding_spec_sha256", "embedding_config_sha256"):
        if not SHA256_RE.fullmatch(str(metadata[key])):
            raise ValueError(f"Text embedding cache {path} has invalid {key}")
    for key in (
        "code_revision",
        "embedding_model_revision",
        "embedding_tokenizer_revision",
        "monitored_model_revision",
        "monitored_tokenizer_revision",
    ):
        if not COMMIT_RE.fullmatch(str(metadata[key])):
            raise ValueError(f"Text embedding cache {path} has invalid {key}")
    if require_clean_code and bool(metadata["code_dirty"]):
        raise ValueError(f"Text embedding cache {path} was produced from a dirty worktree")
    if bool(metadata["normalized"]):
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
            raise ValueError(f"Text embedding cache {path} claims normalization but norms differ")
    group_splits: dict[str, set[str]] = {}
    for group, split in zip(question_ids.tolist(), protocol_splits.tolist()):
        group_splits.setdefault(group, set()).add(split)
    leaking_groups = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaking_groups:
        raise ValueError(
            f"Text embedding cache {path} leaks {len(leaking_groups)} groups across protocol splits"
        )
    if metadata["view"] == "prompt_text":
        prompt_groups: dict[str, set[str]] = {}
        for prompt_hash, group in zip(normalized_text_hashes.tolist(), question_ids.tolist()):
            prompt_groups.setdefault(prompt_hash, set()).add(group)
        duplicate_groups = [
            prompt_hash for prompt_hash, groups in prompt_groups.items() if len(groups) > 1
        ]
        if duplicate_groups:
            raise ValueError(
                f"Text embedding cache {path} repeats normalized prompts across "
                f"{len(duplicate_groups)} different groups"
            )

    return {
        "embeddings": embeddings,
        "labels": labels,
        "example_ids": example_ids,
        "question_ids": question_ids,
        "protocol_splits": protocol_splits,
        "text_sha256": text_hashes,
        "normalized_text_sha256": normalized_text_hashes,
        "embedding_input_sha256": input_hashes,
        "original_token_lengths": token_lengths,
        "truncated": truncated,
        "metadata": metadata,
        "cache_file": str(path),
        "cache_sha256": file_sha256(path),
    }
