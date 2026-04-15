from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np


LAYER_RE = re.compile(r"^(train|eval|test)_layer(-?\d+)\.npz$")


def infer_layers(features_dir: str) -> List[int]:
    layers = []
    for path in Path(features_dir).glob("*_layer*.npz"):
        m = LAYER_RE.match(path.name)
        if m:
            layers.append(int(m.group(2)))
    return sorted(set(layers))


def load_feature_bundle(features_dir: str, split: str, layer: int, cache_suffix: str = "") -> Dict[str, np.ndarray]:
    path = Path(features_dir) / f"{split}_layer{layer}{cache_suffix}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature bundle: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
