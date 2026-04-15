from __future__ import annotations

from pathlib import Path

import numpy as np


META_KEYS = {"labels", "example_ids", "question_ids", "y"}


def transform_bundle(input_path: str | Path, output_path: str | Path, control_type: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    payload = np.load(input_path, allow_pickle=True)
    out = {k: payload[k] for k in payload.files}

    label_key = "labels" if "labels" in out else "y" if "y" in out else None
    if label_key is None:
        raise KeyError("Expected labels or y in bundle")

    feature_keys = [k for k in out.keys() if k not in META_KEYS]

    if control_type == "permute_labels":
        out[label_key] = rng.permutation(out[label_key])
    elif control_type == "shuffle_rows":
        idx = rng.permutation(len(out[label_key]))
        for k in feature_keys:
            out[k] = out[k][idx]
        out[label_key] = out[label_key][idx]
        for meta_key in ["example_ids", "question_ids"]:
            if meta_key in out:
                out[meta_key] = out[meta_key][idx]
    elif control_type == "gaussian_noise":
        for k in feature_keys:
            out[k] = rng.normal(size=out[k].shape).astype(out[k].dtype)
    else:
        raise ValueError(f"Unsupported control_type: {control_type}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **out)
