import json
import sys

import pandas as pd
import yaml

from cli.compute_early_warning_significance import main


def test_early_warning_significance_uses_frozen_prefix_systems(
    tmp_path, monkeypatch
) -> None:
    selection = []
    run_rows = []
    for prefix in (10, 100):
        selection.append(
            {
                "model": "model",
                "source_task": "source",
                "k": 8,
                "prefix_alert_pct": prefix,
                "white_probe": "P8_citm",
                "white_balance_mode": "balanced",
                "white_layer": 2,
                "white_view": f"trajectory_prefix_stack_p{prefix}",
                "black_probe": "B1_text_tfidf",
                "black_balance_mode": "balanced",
                "black_layer": -1,
                "black_view": f"response_prefix_text_p{prefix}",
                "selection_metric": 1.0,
                "selection_rule": "test_rule",
            }
        )
        for seed in (0, 1):
            for access, probe, layer, view, prediction in (
                (
                    "white",
                    "P8_citm",
                    2,
                    f"trajectory_prefix_stack_p{prefix}",
                    True,
                ),
                (
                    "black",
                    "B1_text_tfidf",
                    -1,
                    f"response_prefix_text_p{prefix}",
                    False,
                ),
            ):
                run_id = f"{access}-{prefix}-{seed}"
                prediction_path = tmp_path / f"{run_id}.predictions.jsonl"
                prediction_path.write_text(
                    "\n".join(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "split": "target_test",
                                "example_id": group,
                                "question_id": group,
                                "label": 1,
                                "score": float(prediction),
                                "threshold": 0.5,
                                "predicted_positive": prediction,
                            }
                        )
                        for group in ("g0", "g1")
                    )
                    + "\n",
                    encoding="utf-8",
                )
                run_rows.append(
                    {
                        "run_id": run_id,
                        "status": "ok",
                        "probe": probe,
                        "model": "model",
                        "source_task": "source",
                        "target_task": "target",
                        "k": 8,
                        "balance_mode": "balanced",
                        "layer": layer,
                        "view": view,
                        "seed": seed,
                        "prediction_file": str(prediction_path),
                    }
                )
    (tmp_path / "runs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in run_rows) + "\n",
        encoding="utf-8",
    )
    comparisons = {
        "schema_version": "frontier-primary-comparisons-v1",
        "early_warning_selection": selection,
        "comparisons": [
            {
                "comparison_role": "primary_white_box_gain",
                "common_filters": {
                    "model": "model",
                    "source_task": "source",
                    "target_task": "target",
                    "k": 8,
                },
            }
        ],
    }
    comparisons_path = tmp_path / "comparisons.yaml"
    comparisons_path.write_text(
        yaml.safe_dump(comparisons, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_early_warning_significance",
            "--results_dir",
            str(tmp_path),
            "--comparisons",
            str(comparisons_path),
            "--bootstrap_samples",
            "20",
        ],
    )
    main()

    endpoint = pd.read_csv(
        tmp_path / "early_warning_primary_inference.csv"
    )
    assert endpoint.loc[0, "mean_diff"] == 1.0
    assert endpoint.loc[0, "n_cells"] == 1
    assert (
        endpoint.loc[0, "status"]
        == "confirmatory_hierarchical_inference_complete"
    )
