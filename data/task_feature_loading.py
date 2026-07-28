from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np


LAYER_RE = re.compile(r"^(train|calibration|eval|test)_layer(-?\d+)\.npz$")


def infer_layers(features_dir: str) -> List[int]:
    layers = []
    for path in Path(features_dir).glob("*_layer*.npz"):
        m = LAYER_RE.match(path.name)
        if m:
            layers.append(int(m.group(2)))
    return sorted(set(layers))


def load_feature_bundle(features_dir: str, split: str, layer: int) -> Dict[str, np.ndarray]:
    path = Path(features_dir) / f"{split}_layer{layer}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature bundle: {path}")
    with np.load(path, allow_pickle=False) as data:
        bundle = {k: data[k] for k in data.files}

    required = {"labels", "example_ids", "question_ids"}
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"Feature bundle {path} is missing required arrays: {missing}")
    n_examples = len(bundle["labels"])
    if len(bundle["example_ids"]) != n_examples or len(bundle["question_ids"]) != n_examples:
        raise ValueError(f"Feature bundle {path} has misaligned labels and identifiers")
    for key, value in bundle.items():
        if value.ndim >= 1 and len(value) != n_examples:
            raise ValueError(
                f"Feature bundle {path} has {n_examples} labels but array {key!r} has length {len(value)}"
            )
    return bundle
