from __future__ import annotations

import sys

from cli import run_protocol_multimodel_benchmark as orchestrator


def test_protocol_orchestrator_wires_all_registered_black_box_baselines(
    tmp_path, monkeypatch
) -> None:
    config = {
        "results_dir": str(tmp_path / "results"),
        "run_black_box_baselines": True,
        "run_falsification_suite": True,
        "falsification_registry": "falsification-registry.yaml",
        "falsification_comparisons_file": "falsification-comparisons.yaml",
        "black_box_baselines": [
            "B1_text_tfidf",
            "B2_text_embedding_logistic",
            "B3_llm_judge_zero_shot",
            "B3_llm_judge_few_shot",
            "B4_output_confidence_logistic",
        ],
        "text_embedding_views": [
            "prompt_text",
            "answer_text",
            "transcript_text",
        ],
        "llm_judge_model_lock": "judge-lock.yaml",
        "llm_judge_model_key": "judge-primary",
        "llm_judge_batch_size": 3,
        "llm_judge_views": [
            "prompt_text",
            "answer_text",
            "transcript_text",
        ],
        "llm_judge_modes": ["zero_shot", "few_shot"],
        "task_pairs": [{"source_task": "source", "target_task": "target"}],
        "models": [
            {
                "name": "monitored-model",
                "feature_dirs": {
                    "source": "features/source",
                    "target": "features/target",
                },
                "reference_feature_dir": "features/reference",
                "labeled_data": {"source": "source.jsonl", "target": "target.jsonl"},
                "reference_data": "reference.jsonl",
                "text_embedding_cache_dirs": {
                    "source": "embeddings/source",
                    "target": "embeddings/target",
                },
                "reference_embedding_cache_dir": "embeddings/reference",
                "llm_judge_cache_dir": "judge-cache",
                "falsification_manifests": {
                    "source": "source-falsification.json",
                    "target": "target-falsification.json",
                },
            }
        ],
    }
    commands: list[list[str]] = []
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "load_yaml", lambda _: config)
    monkeypatch.setattr(
        orchestrator, "run_cmd", lambda command: commands.append(command)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_protocol_multimodel_benchmark",
            "--config",
            str(config_path),
        ],
    )

    orchestrator.main()
    assert (
        tmp_path / "results" / "protocol_artifacts" / "execution_manifest.yaml"
    ).exists()

    invoked_modules = [command[2] for command in commands if command[1:2] == ["-m"]]
    for module_name in (
        "cli.run_text_baselines",
        "cli.run_embedding_baselines",
        "cli.run_llm_judge_baselines",
        "cli.run_output_confidence_baselines",
        "cli.evaluate_falsification_slices",
        "cli.compute_falsification_significance",
    ):
        assert invoked_modules.count(module_name) == 1
    judge_command = next(
        command for command in commands if "cli.run_llm_judge_baselines" in command
    )
    assert judge_command[judge_command.index("--modes") + 1] == "zero_shot,few_shot"
    assert judge_command[judge_command.index("--judge_cache_dir") + 1] == "judge-cache"
    confidence_command = next(
        command
        for command in commands
        if "cli.run_output_confidence_baselines" in command
    )
    assert confidence_command[confidence_command.index("--reference_data") + 1] == (
        "reference.jsonl"
    )
    falsification_command = next(
        command
        for command in commands
        if "cli.evaluate_falsification_slices" in command
    )
    manifest_index = falsification_command.index("--manifests")
    assert falsification_command[manifest_index + 1 :] == [
        "source-falsification.json",
        "target-falsification.json",
    ]
    significance_command = next(
        command
        for command in commands
        if "cli.compute_falsification_significance" in command
    )
    assert significance_command[significance_command.index("--comparisons") + 1] == (
        "falsification-comparisons.yaml"
    )

    commands.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_protocol_multimodel_benchmark",
            "--config",
            str(config_path),
            "--selection-only",
            "--results-dir",
            str(tmp_path / "selection-results"),
        ],
    )
    orchestrator.main()
    assert (
        tmp_path
        / "selection-results"
        / "protocol_artifacts"
        / "execution_manifest.yaml"
    ).exists()
    model_runners = [
        command
        for command in commands
        if len(command) > 2
        and command[1:2] == ["-m"]
        and command[2].startswith("cli.run_")
        and command[2]
        not in {
            "cli.run_protocol_multimodel_benchmark",
        }
    ]
    assert all("--selection_only" in command for command in model_runners)
    assert not any(
        "cli.evaluate_falsification_slices" in command for command in commands
    )
    assert any("cli.build_frozen_transfer_report" in command for command in commands)
