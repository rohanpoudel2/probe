from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yaml

from cli import build_geometry_report


def test_geometry_report_ignores_per_example_metadata(tmp_path, monkeypatch) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.savez_compressed(
        feature_dir / "train_layer0.npz",
        answer=np.asarray(
            [[0.0, 0.1], [0.1, 0.0], [1.0, 1.1], [1.1, 1.0]],
            dtype=np.float32,
        ),
        labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        example_ids=np.asarray(["a", "b", "c", "d"]),
        question_ids=np.asarray(["qa", "qb", "qc", "qd"]),
        original_token_counts=np.asarray([10, 11, 12, 13], dtype=np.int64),
        token_spans_json=np.asarray(["{}", "{}", "{}", "{}"]),
        feature_schema_version=np.asarray("3"),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "results_dir": str(tmp_path / "results"),
                "models": [
                    {
                        "name": "model",
                        "feature_dirs": {"task": str(feature_dir)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_geometry_report",
            "--config",
            str(config_path),
        ],
    )

    build_geometry_report.main()

    geometry = pd.read_csv(
        tmp_path / "results" / "task_geometry_summary.csv"
    )
    assert geometry["view"].tolist() == ["answer"]
