from __future__ import annotations

from pathlib import Path

import yaml

from cli.run_frontier_suite import _build_commands


def test_frozen_suite_wires_release_config_and_device(tmp_path) -> None:
    config_path = tmp_path / "frozen.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "protocol_stage": "frozen",
                "results_dir": "results/frozen",
            }
        ),
        encoding="utf-8",
    )
    commands = dict(
        _build_commands(
            config_path,
            controls_config=None,
            ablation_config=None,
            include_controls=False,
            include_ablations=False,
            include_release_steps=True,
            judge_device="mps",
            selection_only=False,
            results_dir_override=None,
        )
    )
    protocol = commands["run_main_protocol"]
    assert protocol[protocol.index("--device") + 1] == "mps"
    claim_tables = commands["build_final_claim_tables"]
    assert claim_tables[claim_tables.index("--config") + 1] == str(config_path)


def test_results_override_reaches_controls_and_release_steps(tmp_path) -> None:
    config_path = tmp_path / "frozen.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "protocol_stage": "frozen",
                "results_dir": "results/original",
            }
        ),
        encoding="utf-8",
    )
    commands = dict(
        _build_commands(
            config_path,
            controls_config=tmp_path / "controls.yaml",
            ablation_config=tmp_path / "ablations.yaml",
            include_controls=True,
            include_ablations=True,
            include_release_steps=True,
            judge_device=None,
            selection_only=False,
            results_dir_override="results/override",
        )
    )
    controls = commands["run_negative_control_suite"]
    assert controls[controls.index("--main_results_dir") + 1] == (
        "results/override"
    )
    claim_tables = commands["build_final_claim_tables"]
    assert claim_tables[claim_tables.index("--results_dir") + 1] == (
        "results/override"
    )
    ablations = commands["run_ablation_suite"]
    assert ablations[ablations.index("--summary-output") + 1] == str(
        Path("results/override") / "ablation_summary.csv"
    )
