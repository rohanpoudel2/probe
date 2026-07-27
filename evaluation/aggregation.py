"""Aggregate result rows into release-ready summary tables."""

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
                    row = json.loads(line)
                    # Root-level falsification evidence is also JSONL but is not
                    # a run summary. Maintained monitor runners always emit this
                    # explicit identity/status contract.
                    if all(key in row for key in ("run_id", "probe", "status")):
                        rows.append(row)
    return pd.DataFrame(rows)
