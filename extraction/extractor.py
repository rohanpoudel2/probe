"""Extract and cache hidden-state activations from language models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _sanitize_model_name(name: str) -> str:
    """Convert model name to a filesystem-safe string."""
    return name.replace("/", "_")


def _get_cache_path(cache_dir: str, model_name: str, dataset_name: str) -> Path:
    return Path(cache_dir) / _sanitize_model_name(model_name) / dataset_name


def _cache_is_complete(out_path: Path, layers: list[int], overwrite: bool) -> bool:
    if overwrite:
        return False
    if not (out_path / "labels.npy").exists():
        return False
    existing_labels = np.load(out_path / "labels.npy")
    layer_files = [out_path / f"layer_{layer}.npy" for layer in layers]
    return len(existing_labels) > 0 and all(f.exists() for f in layer_files)


def _get_transformer_layers(model):
    """Get the list of transformer layers from a supported model."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers
        if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            return inner.decoder.layers
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "h"):
            return model.transformer.h
        if hasattr(model.transformer, "layers"):
            return model.transformer.layers
    raise ValueError(
        f"Cannot find transformer layers for {type(model).__name__}. "
        "Add support in _get_transformer_layers()."
    )


def _compute_pooling_masks(
    attention_mask: torch.Tensor,
    batch_start_positions: list[int] | None = None,
) -> torch.Tensor:
    """Create a token mask used for pooling hidden states."""
    pool_mask = attention_mask.clone().bool()
    if batch_start_positions is None:
        return pool_mask

    for row_idx, start_pos in enumerate(batch_start_positions):
        valid_tokens = int(attention_mask[row_idx].sum().item())
        start_pos = max(0, min(int(start_pos), max(valid_tokens - 1, 0)))
        pool_mask[row_idx, :start_pos] = False
        if not pool_mask[row_idx].any() and valid_tokens > 0:
            pool_mask[row_idx, valid_tokens - 1] = True
    return pool_mask


def _extract_hidden_activations(
    model_name: str,
    texts: list[str],
    layers: list[int],
    batch_size: int,
    max_length: int,
    dtype,
    span_start_tokens: list[int] | None = None,
) -> tuple[dict[int, np.ndarray], list[int]]:
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()

        return hook_fn

    transformer_layers = _get_transformer_layers(model)
    hooks = []
    valid_layers = []
    for layer_idx in layers:
        if layer_idx >= len(transformer_layers):
            print(
                f"Warning: layer {layer_idx} >= model depth {len(transformer_layers)}, skipping"
            )
            continue
        hooks.append(transformer_layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
        valid_layers.append(layer_idx)

    all_activations = {layer: [] for layer in valid_layers}

    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
        batch_texts = texts[batch_start : batch_start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            model(**inputs)

        attention_mask = inputs["attention_mask"]
        batch_spans = None
        if span_start_tokens is not None:
            batch_spans = span_start_tokens[batch_start : batch_start + len(batch_texts)]
        pool_mask = _compute_pooling_masks(attention_mask, batch_spans)
        pool_mask = pool_mask.unsqueeze(-1).float()

        for layer_idx, hidden in captured.items():
            pooled = (hidden.float() * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp_min(1.0)
            all_activations[layer_idx].append(pooled.cpu().numpy())

        captured.clear()
        torch.cuda.empty_cache()

    for hook in hooks:
        hook.remove()

    activations = {
        layer_idx: np.concatenate(acts_list, axis=0)
        for layer_idx, acts_list in all_activations.items()
    }

    del model
    torch.cuda.empty_cache()
    return activations, valid_layers


def _save_cache(
    out_path: Path,
    activations: dict[int, np.ndarray],
    labels: np.ndarray,
    model_name: str,
    dataset_name: str,
    layers: list[int],
    max_length: int,
    dtype,
):
    out_path.mkdir(parents=True, exist_ok=True)
    for layer_idx, acts in activations.items():
        np.save(out_path / f"layer_{layer_idx}.npy", acts)
        print(f"  Saved layer {layer_idx}: shape {acts.shape}")

    np.save(out_path / "labels.npy", labels)

    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_samples": int(len(labels)),
        "layers": layers,
        "max_length": max_length,
        "dtype": str(dtype),
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Extraction complete. Cached at {out_path}")


def extract_and_cache(
    model_name: str,
    texts: list[str],
    labels: np.ndarray,
    layers: list[int],
    cache_dir: str,
    dataset_name: str,
    batch_size: int = 16,
    max_length: int = 512,
    dtype=torch.bfloat16,
    overwrite: bool = False,
):
    """Extract hidden-state activations and save mean-pooled activations to disk."""
    out_path = _get_cache_path(cache_dir, model_name, dataset_name)
    if _cache_is_complete(out_path, layers, overwrite):
        existing_labels = np.load(out_path / "labels.npy")
        if len(existing_labels) == len(labels):
            print(f"Cache already exists at {out_path}, skipping extraction.")
            return

    activations, valid_layers = _extract_hidden_activations(
        model_name=model_name,
        texts=texts,
        layers=layers,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
    )
    _save_cache(out_path, activations, labels, model_name, dataset_name, valid_layers, max_length, dtype)


def extract_and_cache_with_span_mask(
    model_name: str,
    texts: list[str],
    labels: np.ndarray,
    layers: list[int],
    cache_dir: str,
    dataset_name: str,
    span_start_tokens: list[int],
    batch_size: int = 16,
    max_length: int = 512,
    dtype=torch.bfloat16,
    overwrite: bool = False,
):
    """Extract activations pooled only over tokens at or after a start position."""
    if len(span_start_tokens) != len(texts):
        raise ValueError("span_start_tokens must have the same length as texts")

    out_path = _get_cache_path(cache_dir, model_name, dataset_name)
    if _cache_is_complete(out_path, layers, overwrite):
        existing_labels = np.load(out_path / "labels.npy")
        if len(existing_labels) == len(labels):
            print(f"Cache already exists at {out_path}, skipping extraction.")
            return

    activations, valid_layers = _extract_hidden_activations(
        model_name=model_name,
        texts=texts,
        layers=layers,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
        span_start_tokens=span_start_tokens,
    )
    _save_cache(out_path, activations, labels, model_name, dataset_name, valid_layers, max_length, dtype)


def load_cached_activations(
    cache_dir: str, model_name: str, dataset_name: str, layer: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached activations and labels for a specific layer."""
    path = _get_cache_path(cache_dir, model_name, dataset_name)
    acts = np.load(path / f"layer_{layer}.npy")
    labels = np.load(path / "labels.npy")
    return acts, labels
