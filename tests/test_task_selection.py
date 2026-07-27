import pandas as pd
import pytest

from evaluation.task_selection import (
    apply_frozen_selection,
    select_frozen_source_systems,
    select_primary_source_systems,
)


def test_layer_view_is_selected_at_fixed_k_and_reused_across_k_and_targets() -> None:
    rows = []
    for target in ("t1", "t2"):
        for k in (1, 8):
            rows.extend(
                [
                    {
                        "model": "m",
                        "source_task": "s",
                        "target_task": target,
                        "probe": "p",
                        "balance_mode": "balanced",
                        "layer": 1,
                        "view": "answer",
                        "k": k,
                        "eval_tpr_at_reference_alert_budget_mean": 0.9 if k == 1 else 0.2,
                    },
                    {
                        "model": "m",
                        "source_task": "s",
                        "target_task": target,
                        "probe": "p",
                        "balance_mode": "balanced",
                        "layer": 2,
                        "view": "answer",
                        "k": k,
                        "eval_tpr_at_reference_alert_budget_mean": 0.1 if k == 1 else 0.8,
                    },
                ]
            )
    summary = pd.DataFrame(rows)
    selected = select_frozen_source_systems(summary, selection_k=8)
    assert selected["layer"].tolist() == [2]
    report = apply_frozen_selection(summary, selected)
    assert set(report["k"]) == {1, 8}
    assert set(report["target_task"]) == {"t1", "t2"}
    assert set(report["layer"]) == {2}


def test_primary_selection_jointly_freezes_family_layer_and_view() -> None:
    rows = []
    systems = [
        ("P1_logistic", "balanced", 1, "answer", 0.70),
        ("P2_mass_mean", "balanced", 5, "reasoning", 0.85),
        ("B1_text_tfidf", "balanced", -1, "answer_text", 0.65),
        ("B3_llm_judge_zero_shot", "none", -3, "transcript_text", 0.80),
    ]
    for target_task, transfer_bonus in (("t1", 0.0), ("t2", 10.0)):
        for probe, balance_mode, layer, view, source_score in systems:
            rows.append(
                {
                    "model": "m",
                    "source_task": "s",
                    "target_task": target_task,
                    "probe": probe,
                    "balance_mode": balance_mode,
                    "layer": layer,
                    "view": view,
                    "k": 8,
                    "eval_tpr_at_reference_alert_budget_mean": source_score,
                    # Deliberately favors P1 on a target; primary selection
                    # must never inspect this column.
                    "transfer_tpr_at_1pct_reference_alert_budget_mean": (
                        transfer_bonus if probe == "P1_logistic" else 0.0
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    selected = select_primary_source_systems(summary, selection_k=8)

    white_box = selected[selected["access_regime"] == "white_box"].iloc[0]
    black_box = selected[selected["access_regime"] == "black_box"].iloc[0]
    assert (white_box["probe"], white_box["layer"], white_box["view"]) == (
        "P2_mass_mean",
        5,
        "reasoning",
    )
    assert (
        black_box["probe"],
        black_box["balance_mode"],
        black_box["view"],
    ) == ("B3_llm_judge_zero_shot", "none", "transcript_text")


def test_primary_selection_uses_deterministic_tie_break() -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "m",
                "source_task": "s",
                "target_task": "t",
                "probe": probe,
                "balance_mode": "balanced",
                "layer": layer,
                "view": "answer",
                "k": 4,
                "eval_tpr_at_reference_alert_budget_mean": 0.5,
            }
            for probe, layer in (
                ("P2_mass_mean", 2),
                ("P1_logistic", 1),
                ("B2_text_embedding_logistic", -2),
                ("B1_text_tfidf", -1),
            )
        ]
    )
    selected = select_primary_source_systems(summary, selection_k=4)
    assert selected.set_index("access_regime")["probe"].to_dict() == {
        "black_box": "B1_text_tfidf",
        "white_box": "P1_logistic",
    }


def test_primary_selection_requires_chosen_budget_for_every_access_regime() -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "m",
                "source_task": "s",
                "target_task": "t",
                "probe": "P1_logistic",
                "balance_mode": "balanced",
                "layer": 1,
                "view": "answer",
                "k": 8,
                "eval_tpr_at_reference_alert_budget_mean": 0.5,
            },
            {
                "model": "m",
                "source_task": "s",
                "target_task": "t",
                "probe": "B1_text_tfidf",
                "balance_mode": "balanced",
                "layer": -1,
                "view": "answer_text",
                "k": 4,
                "eval_tpr_at_reference_alert_budget_mean": 0.5,
            },
        ]
    )
    with pytest.raises(ValueError, match="not available for every access regime"):
        select_primary_source_systems(summary, selection_k=8)


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf")])
def test_primary_selection_requires_finite_metric_per_access_regime(
    invalid_score: float,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "m",
                "source_task": "s",
                "target_task": "t",
                "probe": probe,
                "balance_mode": "balanced",
                "layer": layer,
                "view": view,
                "k": 8,
                "eval_tpr_at_reference_alert_budget_mean": score,
            }
            for probe, layer, view, score in (
                ("P1_logistic", 1, "answer", 0.5),
                ("B1_text_tfidf", -1, "answer_text", invalid_score),
            )
        ]
    )
    with pytest.raises(ValueError, match="No finite"):
        select_primary_source_systems(summary, selection_k=8)
