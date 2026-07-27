from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _write_bundle(
    directory: Path,
    split: str,
    labels: np.ndarray,
    *,
    offset: float = 0.0,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int64)
    index = np.arange(len(labels), dtype=np.float32)
    answer = np.column_stack(
        [
            labels.astype(np.float32) * 2.0 + offset + index * 0.001,
            labels.astype(np.float32) + offset - index * 0.001,
        ]
    )
    matched_behavior_groups = set(np.unique(labels).tolist()) == {0, 1}
    np.savez_compressed(
        directory / f"{split}_layer0.npz",
        answer=answer,
        full_text=answer + 0.05,
        labels=labels,
        example_ids=np.asarray([f"{split}-e{i}" for i in range(len(labels))]),
        question_ids=np.asarray(
            [
                f"{split}-g{i // 2 if matched_behavior_groups else i}"
                for i in range(len(labels))
            ]
        ),
    )


def test_non_model_protocol_runs_through_all_postprocessors(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    reference = tmp_path / "reference"
    behavior_labels = np.asarray([0, 1] * 5, dtype=np.int64)
    for split in ("train", "eval", "test"):
        _write_bundle(source, split, behavior_labels)
        _write_bundle(target, split, behavior_labels, offset=0.1)
    _write_bundle(
        reference,
        "calibration",
        np.zeros(100, dtype=np.int64),
        offset=-2.0,
    )
    _write_bundle(
        reference,
        "test",
        np.zeros(100, dtype=np.int64),
        offset=-2.0,
    )

    results = tmp_path / "results"
    config = {
        "execution_mode": "confirmatory",
        "results_dir": str(results),
        "overwrite": False,
        "bootstrap_samples": 20,
        "views": "answer,full_text",
        "layers": "all",
        "probes": "P2_mass_mean",
        "k_values": "1",
        "seeds": 2,
        "balance_modes": "balanced",
        "max_reference_alert_rate": 0.01,
        "min_reference_groups": 100,
        "run_black_box_baselines": False,
        "run_falsification_suite": False,
        "calibration_pairs": [
            {"source_task": "source", "target_task": "source"}
        ],
        "transfer_pairs": [
            {"source_task": "source", "target_task": "target"}
        ],
        "models": [
            {
                "name": "synthetic-model",
                "feature_dirs": {
                    "source": str(source),
                    "target": str(target),
                },
                "reference_feature_dir": str(reference),
            }
        ],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.run_protocol_multimodel_benchmark",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    expected = {
        "task_summary.csv",
        "task_primary_transfer_report.csv",
        "task_cross_model_table.csv",
        "task_same_task_calibration.csv",
        "task_cross_task_transfer.csv",
        "task_geometry_summary.csv",
        "cross_model_transfer.png",
        "label_budget_scaling.png",
    }
    assert expected.issubset({path.name for path in results.iterdir()})
    cross_model = pd.read_csv(results / "task_cross_model_table.csv")
    assert len(cross_model) == 2
    assert set(cross_model["access_regime"]) == {"white_box"}
    geometry = pd.read_csv(results / "task_geometry_summary.csv")
    assert set(geometry["view"]) == {"answer", "full_text"}
