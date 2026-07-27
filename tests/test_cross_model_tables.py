from __future__ import annotations

import sys

import pandas as pd
import pytest

from cli import build_cross_model_tables


def _row(model: str, access: str, probe: str, k: int, value: float) -> dict:
    return {
        "model": model,
        "source_task": "source",
        "target_task": "target",
        "access_regime": access,
        "probe": probe,
        "k": k,
        "transfer_tpr_at_1pct_reference_alert_budget_mean": value,
    }


def test_cross_model_table_preserves_access_regime_and_k(
    tmp_path, monkeypatch
) -> None:
    report = pd.DataFrame(
        [
            _row("m1", "white_box", "P1", 1, 0.2),
            _row("m1", "white_box", "P1", 8, 0.8),
            _row("m1", "black_box", "B1", 1, 0.1),
            _row("m1", "black_box", "B1", 8, 0.4),
            _row("m2", "white_box", "P2", 1, 0.3),
            _row("m2", "white_box", "P2", 8, 0.7),
            _row("m2", "black_box", "B2", 1, 0.2),
            _row("m2", "black_box", "B2", 8, 0.5),
        ]
    )
    report.to_csv(tmp_path / "task_primary_transfer_report.csv", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_cross_model_tables", "--results_dir", str(tmp_path)],
    )

    build_cross_model_tables.main()

    table = pd.read_csv(tmp_path / "task_cross_model_table.csv")
    assert len(table) == 4
    assert set(table["access_regime"]) == {"white_box", "black_box"}
    assert set(table["k"]) == {1, 8}
    row = table[
        (table["access_regime"] == "white_box") & (table["k"] == 1)
    ].iloc[0]
    assert row["m1"] == pytest.approx(0.2)
    assert row["m2"] == pytest.approx(0.3)


def test_cross_model_table_rejects_duplicate_primary_identity(
    tmp_path, monkeypatch
) -> None:
    report = pd.DataFrame(
        [
            _row("m1", "white_box", "P1", 8, 0.8),
            _row("m1", "white_box", "P2", 8, 0.9),
        ]
    )
    report.to_csv(tmp_path / "task_primary_transfer_report.csv", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_cross_model_tables", "--results_dir", str(tmp_path)],
    )

    with pytest.raises(ValueError, match="multiple source-selected systems"):
        build_cross_model_tables.main()
