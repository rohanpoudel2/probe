from __future__ import annotations

from pathlib import Path

import numpy as np


META_KEYS = {"labels", "example_ids", "question_ids", "y"}


def transform_bundle(input_path: str | Path, output_path: str | Path, control_type: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    payload = np.load(input_path, allow_pickle=False)
    out = {k: payload[k] for k in payload.files}

    label_key = "labels" if "labels" in out else "y" if "y" in out else None
    if label_key is None:
        raise KeyError("Expected labels or y in bundle")

    n_examples = len(out[label_key])
    feature_keys = [
        key
        for key, value in out.items()
        if key not in META_KEYS
        and value.ndim >= 1
        and len(value) == n_examples
        and np.issubdtype(value.dtype, np.number)
    ]
    if not feature_keys:
        raise ValueError("Bundle contains no aligned numeric feature arrays")

    if control_type == "permute_labels":
        out[label_key] = rng.permutation(out[label_key])
    elif control_type == "permute_features":
        idx = rng.permutation(len(out[label_key]))
        for k in feature_keys:
            out[k] = out[k][idx]
    elif control_type == "gaussian_noise":
        for k in feature_keys:
            values = np.asarray(out[k], dtype=float)
            mean = values.mean(axis=0, keepdims=True)
            std = values.std(axis=0, keepdims=True)
            std = np.where(std > 0.0, std, 1.0)
            out[k] = rng.normal(loc=mean, scale=std, size=values.shape).astype(out[k].dtype)
    elif control_type == "shuffle_rows":
        raise ValueError(
            "shuffle_rows was removed because shuffling features, labels, and IDs together preserves the task. "
            "Use permute_features to destroy feature-label alignment."
        )
    else:
        raise ValueError(f"Unsupported control_type: {control_type}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **out)
