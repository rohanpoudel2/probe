"""Extract and cache hidden-state activations from language models.

This is the GPU-heavy phase. Run once per (model, dataset) pair, then all
probe training reads from the cached .npy files on CPU.
"""

import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    """Pick the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device):
    """Pick a safe dtype for the device. MPS doesn't support bfloat16."""
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16


def empty_cache():
    """Free accelerator memory if available."""
    if torch.cuda.is_available():
        empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _sanitize_model_name(name: str) -> str:
    """Convert model name to filesystem-safe string."""
    return name.replace("/", "_")


def _get_cache_path(cache_dir: str, model_name: str, dataset_name: str) -> Path:
    return Path(cache_dir) / _sanitize_model_name(model_name) / dataset_name


def extract_and_cache(
    model_name: str,
    texts: list[str],
    labels: np.ndarray,
    layers: list[int],
    cache_dir: str,
    dataset_name: str,
    batch_size: int = 16,
    max_length: int = 512,
):
    """Extract hidden-state activations and save to disk.

    For each specified layer, saves a .npy file of shape [N, hidden_dim]
    containing mean-pooled activations over the sequence length.

    Args:
        model_name: HuggingFace model identifier.
        texts: List of input texts.
        labels: Array of labels, shape [N].
        layers: Which transformer layers to extract from (0-indexed).
        cache_dir: Root directory for activation caches.
        dataset_name: Name for this dataset (used in cache path).
        batch_size: Batch size for forward passes.
        max_length: Max token length.
    """
    out_path = _get_cache_path(cache_dir, model_name, dataset_name)
    out_path.mkdir(parents=True, exist_ok=True)

    # Check if already cached
    if (out_path / "labels.npy").exists():
        existing_labels = np.load(out_path / "labels.npy")
        if len(existing_labels) == len(labels):
            layer_files = [out_path / f"layer_{l}.npy" for l in layers]
            if all(f.exists() for f in layer_files):
                print(f"Cache already exists at {out_path}, skipping extraction.")
                return

    device = get_device()
    dtype = get_dtype(device)
    print(f"Loading model {model_name} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device.type == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set up hooks to capture hidden states at target layers
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is typically a tuple; first element is the hidden state
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook_fn

    # Register hooks on the target layers
    transformer_layers = _get_transformer_layers(model)
    hooks = []
    for layer_idx in layers:
        if layer_idx >= len(transformer_layers):
            print(f"Warning: layer {layer_idx} >= model depth {len(transformer_layers)}, skipping")
            continue
        h = transformer_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Accumulate activations
    all_activations = {l: [] for l in layers if l < len(transformer_layers)}

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            model(**inputs)

        attention_mask = inputs["attention_mask"]  # [B, seq_len]

        for layer_idx, hidden in captured.items():
            # Mean-pool over sequence length (masked)
            mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1)  # [B, D]
            all_activations[layer_idx].append(pooled.cpu().numpy())

        captured.clear()
        empty_cache()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save to disk
    for layer_idx, acts_list in all_activations.items():
        acts = np.concatenate(acts_list, axis=0)  # [N, D]
        np.save(out_path / f"layer_{layer_idx}.npy", acts)
        print(f"  Saved layer {layer_idx}: shape {acts.shape}")

    np.save(out_path / "labels.npy", labels)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_samples": len(texts),
        "layers": [l for l in layers if l < len(transformer_layers)],
        "max_length": max_length,
        "dtype": str(dtype),
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Extraction complete. Cached at {out_path}")

    # Free model from GPU
    del model
    torch.cuda.empty_cache()


def _get_transformer_layers(model):
    """Get the list of transformer layer modules from a model.

    Handles different model architectures (Qwen, Gemma, LLaMA-style).
    """
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


def load_cached_activations(
    cache_dir: str, model_name: str, dataset_name: str, layer: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached activations and labels for a specific layer.

    Returns:
        (activations, labels) — shapes [N, D] and [N].
    """
    path = _get_cache_path(cache_dir, model_name, dataset_name)
    acts = np.load(path / f"layer_{layer}.npy")
    labels = np.load(path / "labels.npy")
    return acts, labels
