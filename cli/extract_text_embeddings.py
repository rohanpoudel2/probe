from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from data.rollout_schema import canonical_json
from data.text_embedding_cache import (
    TEXT_EMBEDDING_SCHEMA_VERSION,
    atomic_save_text_embedding_cache,
)
from data.text_views import (
    ALLOWED_TEXT_VIEWS,
    examples_to_text_arrays,
    monitored_model_identity,
)
from tasks import TASK_REGISTRY


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _git_state() -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError) as err:
        raise RuntimeError("Text embedding extraction requires Git provenance") from err
    return revision, dirty


def _load_embedding_spec(config_path: Path, key: str) -> tuple[dict[str, Any], str, str]:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes) or {}
    if config.get("schema_version") != "text-embedding-model-lock-v1":
        raise ValueError("Unsupported text embedding model-lock schema")
    models = config.get("models") or {}
    if key not in models:
        raise ValueError(f"Unknown embedding model key {key!r}")
    spec = dict(models[key])
    required = {
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "pooling",
        "padding_side",
        "normalize",
        "max_length",
        "instruction",
        "instruction_format",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"Embedding model lock {key!r} is missing {missing}")
    for field in ("model_revision", "tokenizer_revision"):
        if not COMMIT_RE.fullmatch(str(spec[field])):
            raise ValueError(f"Embedding model lock {key!r} has an unpinned {field}")
    if spec["pooling"] not in {"last", "mean"}:
        raise ValueError("Embedding pooling must be last or mean")
    if spec["padding_side"] not in {"left", "right"}:
        raise ValueError("Embedding padding_side must be left or right")
    if not isinstance(spec["normalize"], bool) or int(spec["max_length"]) < 1:
        raise ValueError("Embedding normalize/max_length settings are invalid")
    if "{instruction}" not in spec["instruction_format"] or "{text}" not in spec[
        "instruction_format"
    ]:
        raise ValueError("instruction_format must contain {instruction} and {text}")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    spec_hash = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()
    return spec, config_hash, spec_hash


def render_embedding_input(text: str, spec: dict[str, Any]) -> str:
    return str(spec["instruction_format"]).format(
        instruction=spec["instruction"], text=text
    )


def pool_hidden_states(last_hidden_state, attention_mask, mode: str):
    import torch

    mask = attention_mask.to(dtype=last_hidden_state.dtype)
    if mask.ndim != 2 or last_hidden_state.ndim != 3:
        raise ValueError("Hidden states and attention mask have invalid dimensions")
    if last_hidden_state.shape[:2] != mask.shape or torch.any(mask.sum(dim=1) <= 0):
        raise ValueError("Hidden states and attention mask are not aligned")
    if mode == "mean":
        return (last_hidden_state * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        )
    if mode == "last":
        positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        last_positions = (positions * mask.to(dtype=torch.long)).argmax(dim=1)
        batch = torch.arange(mask.shape[0], device=mask.device)
        return last_hidden_state[batch, last_positions]
    raise ValueError(f"Unknown pooling mode: {mode}")


def _encode(
    texts: list[str],
    *,
    tokenizer,
    model,
    spec: dict[str, Any],
    batch_size: int,
    allow_truncation: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    max_length = int(spec["max_length"])
    model_device = next(model.parameters()).device
    embeddings: list[np.ndarray] = []
    lengths: list[int] = []
    truncated: list[bool] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        untruncated = tokenizer(
            batch_texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        batch_lengths = [len(ids) for ids in untruncated]
        batch_truncated = [length > max_length for length in batch_lengths]
        if any(batch_truncated) and not allow_truncation:
            indices = [start + index for index, value in enumerate(batch_truncated) if value]
            raise ValueError(
                f"Embedding inputs {indices[:5]} exceed max_length={max_length}; "
                "shorten data or explicitly pass --allow_truncation"
            )
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("Embedding model output lacks last_hidden_state")
        pooled = pool_hidden_states(hidden, encoded["attention_mask"], str(spec["pooling"]))
        if spec["normalize"]:
            pooled = functional.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.detach().float().cpu().numpy())
        lengths.extend(batch_lengths)
        truncated.extend(batch_truncated)
    return (
        np.concatenate(embeddings, axis=0).astype(np.float32, copy=False),
        np.asarray(lengths, dtype=np.int64),
        np.asarray(truncated, dtype=bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute one reusable pinned-transformer embedding cache per text view"
    )
    parser.add_argument("--task", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--data", required=True)
    parser.add_argument("--embedding_config", required=True)
    parser.add_argument("--embedding_model_key", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--views", default="prompt_text,answer_text,transcript_text")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--allow_truncation", action="store_true")
    parser.add_argument("--allow_dirty_code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto|cpu|cuda|cuda:N|mps",
    )
    args = parser.parse_args()

    views = [value.strip() for value in args.views.split(",") if value.strip()]
    if not views or not set(views).issubset(ALLOWED_TEXT_VIEWS):
        raise ValueError(f"Text views must be chosen from {sorted(ALLOWED_TEXT_VIEWS)}")
    data_path = Path(args.data)
    config_path = Path(args.embedding_config)
    spec, config_hash, spec_hash = _load_embedding_spec(
        config_path, args.embedding_model_key
    )
    code_revision, code_dirty = _git_state()
    if code_dirty and not args.allow_dirty_code:
        raise RuntimeError(
            "Refusing text embedding extraction from a dirty worktree; commit the protocol "
            "or pass --allow_dirty_code for a non-final pilot"
        )

    task = TASK_REGISTRY[args.task]()
    examples = task.load(args.data)
    if any(
        example.metadata.get("data_origin") != "on_policy_generation"
        or example.metadata.get("generated_by_model") is not True
        for example in examples
    ):
        raise ValueError("Text embeddings require only on-policy model outputs")
    identity = monitored_model_identity(examples)
    dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    split_values = {str(example.metadata.get("protocol_split") or "") for example in examples}
    if not split_values.issubset({"train", "calibration", "eval", "test"}):
        raise ValueError(f"Dataset contains invalid protocol splits: {sorted(split_values)}")

    from transformers import AutoModel, AutoTokenizer
    from cli.common import resolve_torch_device

    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"],
        revision=spec["tokenizer_revision"],
        padding_side=spec["padding_side"],
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Embedding tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    resolved_device = resolve_torch_device(args.device)
    model_kwargs = {
        "revision": spec["model_revision"],
        "torch_dtype": "auto",
    }
    if resolved_device == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModel.from_pretrained(
        spec["model_id"],
        **model_kwargs,
    )
    if resolved_device != "auto":
        model = model.to(resolved_device)
    model.eval()

    output_dir = Path(args.output_dir)
    completed = 0
    for view in views:
        output_path = output_dir / f"{args.task}__{view}.npz"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing cache {output_path}")
        arrays = examples_to_text_arrays(examples, view)
        raw_texts = arrays.pop("texts").tolist()
        rendered = [render_embedding_input(text, spec) for text in raw_texts]
        embeddings, token_lengths, truncated = _encode(
            rendered,
            tokenizer=tokenizer,
            model=model,
            spec=spec,
            batch_size=args.batch_size,
            allow_truncation=args.allow_truncation,
        )
        arrays.update(
            {
                "embeddings": embeddings,
                "text_sha256": np.asarray(
                    [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in raw_texts]
                ),
                "normalized_text_sha256": np.asarray(
                    [
                        hashlib.sha256(
                            " ".join(text.casefold().split()).encode("utf-8")
                        ).hexdigest()
                        for text in raw_texts
                    ]
                ),
                "embedding_input_sha256": np.asarray(
                    [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in rendered]
                ),
                "original_token_lengths": token_lengths,
                "truncated": truncated,
            }
        )
        metadata = {
            "schema_version": TEXT_EMBEDDING_SCHEMA_VERSION,
            "task_name": args.task,
            "view": view,
            "dataset_sha256": dataset_hash,
            "code_revision": code_revision,
            "code_dirty": code_dirty,
            "embedding_model_id": spec["model_id"],
            "embedding_model_revision": spec["model_revision"],
            "embedding_tokenizer_revision": spec["tokenizer_revision"],
            "embedding_spec_sha256": spec_hash,
            "embedding_config_sha256": config_hash,
            "pooling": spec["pooling"],
            "padding_side": spec["padding_side"],
            "normalized": spec["normalize"],
            "max_length": int(spec["max_length"]),
            "instruction": spec["instruction"],
            "instruction_format": spec["instruction_format"],
            **identity,
        }
        atomic_save_text_embedding_cache(output_path, arrays=arrays, metadata=metadata)
        completed += 1
        print(
            json.dumps(
                {
                    "saved": str(output_path),
                    "n_examples": len(embeddings),
                    "embedding_dimension": int(embeddings.shape[1]),
                    "n_truncated": int(np.sum(truncated)),
                    "embedding_spec_sha256": spec_hash,
                },
                sort_keys=True,
            )
        )
    print(f"completed {completed} reusable text embedding caches")


if __name__ == "__main__":
    main()
