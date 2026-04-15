"""Aggregate result rows into paper ready summary tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def collect_results(results_dir: str) -> pd.DataFrame:
    results_path = Path(results_dir)
    rows = []
    for f in sorted(results_path.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)
