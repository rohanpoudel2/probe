from __future__ import annotations

import pandas as pd
import pytest

from cli.compute_task_significance import _validate_primary_source_selection


def _selection() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "m",
                "source_task": "s",
                "access_regime": "white_box",
                "probe": "P2_mass_mean",
                "balance_mode": "balanced",
                "layer": 5,
                "view": "reasoning",
            },
            {
                "model": "m",
                "source_task": "s",
                "access_regime": "black_box",
                "probe": "B3_llm_judge_zero_shot",
                "balance_mode": "none",
                "layer": -3,
                "view": "transcript_text",
            },
        ]
    )


def test_primary_comparison_must_match_source_selected_identities() -> None:
    comparison = {"comparison_role": "primary_white_box_gain"}
    white_box = {
        "model": "m",
        "source_task": "s",
        "probe": "P2_mass_mean",
        "balance_mode": "balanced",
        "layer": 5,
        "view": "reasoning",
    }
    black_box = {
        "model": "m",
        "source_task": "s",
        "probe": "B3_llm_judge_zero_shot",
        "balance_mode": "none",
        "layer": -3,
        "view": "transcript_text",
    }
    _validate_primary_source_selection(
        comparison, white_box, black_box, _selection(), "primary"
    )

    with pytest.raises(ValueError, match="does not match"):
        _validate_primary_source_selection(
            comparison,
            {**white_box, "layer": 6},
            black_box,
            _selection(),
            "primary",
        )
