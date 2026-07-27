from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cli.common import load_yaml
from cli.freeze_protocol_from_selection import (
    build_falsification_comparisons,
    build_primary_comparisons,
    validate_selection_evidence,
)
from data.falsification import SHIFT_AXES, load_falsification_registry
from evaluation.task_selection import select_primary_source_systems


def _selected(config: dict) -> pd.DataFrame:
    rows = []
    for model in config["models"]:
        for regime, probe, layer, view in (
            ("white_box", "P1_logistic", 15, "answer"),
            (
                "black_box",
                "B2_text_embedding_logistic",
                -2,
                "transcript_text",
            ),
        ):
            rows.append(
                {
                    "model": model["name"],
                    "source_task": "sycophancy",
                    "access_regime": regime,
                    "probe": probe,
                    "balance_mode": (
                        "balanced" if regime == "white_box" else "none"
                    ),
                    "layer": layer,
                    "view": view,
                }
            )
    return pd.DataFrame(rows)


def test_freezer_covers_every_model_pair_budget_and_falsification_axis() -> None:
    config = load_yaml("experiments/protocol/main_research_manifest.yaml")
    selected = _selected(config)
    primary = build_primary_comparisons(config, selected)
    assert len(primary["comparisons"]) == 4 * 3 * 6

    registry, _ = load_falsification_registry(
        Path("experiments/protocol/falsification_registry.yaml")
    )
    falsification = build_falsification_comparisons(
        config, selected, registry, selection_k=8
    )
    assert {
        comparison["system_b"]["balance_mode"]
        for comparison in falsification["comparisons"]
    } == {"none"}
    covered = {
        (row["common_filters"]["model"], row["slice"]["axis"])
        for row in falsification["comparisons"]
        if row["slice"]["type"] == "shift"
    }
    assert covered == {
        (model["name"], axis)
        for model in config["models"]
        for axis in SHIFT_AXES
    }
    hard_metrics = {
        row["metric"]
        for row in falsification["comparisons"]
        if row["slice"]["type"] == "matched_hard_negative"
    }
    assert hard_metrics == {"hard_negative_fpr", "pairwise_order_accuracy"}


def test_freezer_rejects_selection_runs_that_scored_test_data(tmp_path) -> None:
    config = {
        "execution_mode": "selection",
        "seeds": 1,
        "probes": "P1_logistic",
        "run_black_box_baselines": True,
        "black_box_baselines": ["B1_text_tfidf"],
        "text_embedding_views": ["answer"],
        "task_pairs": [{"source_task": "source", "target_task": "target"}],
        "models": [{"name": "model"}],
    }
    rows = []
    for probe, layer in (
        ("P1_logistic", 1),
        ("B1_text_tfidf", -1),
    ):
        rows.append(
            {
                "run_id": f"model__source__target__{probe}__selection",
                "status": "ok",
                "execution_mode": "selection",
                "model": "model",
                "source_task": "source",
                "target_task": "target",
                "probe": probe,
                "balance_mode": "balanced",
                "layer": layer,
                "view": "answer",
                "k": 8,
                "eval_tpr_at_reference_alert_budget": 0.7,
                "test_tpr_at_reference_alert_budget": float("nan"),
                "transfer_tpr_at_reference_alert_budget": float("nan"),
            }
        )
    with (tmp_path / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = pd.DataFrame(
        [
            {
                "model": row["model"],
                "source_task": row["source_task"],
                "target_task": row["target_task"],
                "probe": row["probe"],
                "balance_mode": row["balance_mode"],
                "layer": row["layer"],
                "view": row["view"],
                "k": row["k"],
                "eval_tpr_at_reference_alert_budget_mean": row[
                    "eval_tpr_at_reference_alert_budget"
                ],
                "eval_tpr_at_reference_alert_budget_n_seeds": 1,
            }
            for row in rows
        ]
    )
    summary.to_csv(tmp_path / "task_summary.csv", index=False)
    selected = select_primary_source_systems(summary, selection_k=8)
    validate_selection_evidence(
        config,
        tmp_path,
        selected,
        selection_k=8,
    )

    contaminated = rows.copy()
    contaminated[0] = {
        **contaminated[0],
        "test_tpr_at_reference_alert_budget": 0.5,
    }
    with (tmp_path / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in contaminated:
            handle.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="behavior-test values"):
        validate_selection_evidence(
            config,
            tmp_path,
            selected,
            selection_k=8,
        )
