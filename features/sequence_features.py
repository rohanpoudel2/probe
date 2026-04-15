from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class SequenceFeatureBundle:
    full: np.ndarray
    spans: Dict[str, np.ndarray]
    token_states: Optional[np.ndarray] = None
    metadata: Optional[dict] = None
