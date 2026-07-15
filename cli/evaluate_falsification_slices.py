from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.falsification import (
    SHIFT_AXES,
    file_sha256,
    load_falsification_evaluation_manifest,
    load_falsification_registry,
)
from evaluation.aggregation import collect_results
from evaluation.metrics import compute_auprc, compute_auroc


def _prediction_path(path_value: str, results_dir: Path) -> Path:
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([results_dir / path, results_dir.parent / path])
    resolved = next((candidate for candidate in candidates if candidate.exists()), None)
    if resolved is None:
        raise FileNotFoundError(f"Missing prediction artifact {path_value}")
    return resolved


def _load_predictions(path_value: str, results_dir: Path, split: str) -> pd.DataFrame:
    predictions = pd.read_json(_prediction_path(path_value, results_dir), lines=True)
    required = {
        "split",
        "example_id",
        "question_id",
        "label",
        "score",
        "predicted_positive",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction artifact lacks fields {missing}")
    predictions = predictions[predictions["split"] == split].copy()
    if predictions.empty:
        raise ValueError(f"Prediction artifact has no {split} rows")
    if predictions["example_id"].astype(str).duplicated().any():
        raise ValueError(f"Prediction artifact duplicates example IDs in {split}")
    return predictions


def _slice_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    labels = rows["label"].to_numpy(dtype=np.int64)
    scores = rows["score"].to_numpy(dtype=float)
    predicted = rows["predicted_positive"].astype(bool).to_numpy()
    positive = labels == 1
    negative = labels == 0
    return {
        "n_examples": int(len(rows)),
        "n_groups": int(rows["question_id"].astype(str).nunique()),
        "n_positive": int(np.sum(positive)),
        "n_negative": int(np.sum(negative)),
        "auroc": compute_auroc(labels, scores),
        "auprc": compute_auprc(labels, scores),
        "tpr_at_frozen_threshold": (
            float(np.mean(predicted[positive])) if np.any(positive) else float("nan")
        ),
        "fpr_at_frozen_threshold": (
            float(np.mean(predicted[negative])) if np.any(negative) else float("nan")
        ),
        "mean_positive_score": (
            float(np.mean(scores[positive])) if np.any(positive) else float("nan")
        ),
        "mean_negative_score": (
            float(np.mean(scores[negative])) if np.any(negative) else float("nan")
        ),
    }


def _run_metadata(
    row: pd.Series, manifest: dict[str, Any], split: str
) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "probe": row["probe"],
        "model": row["model"],
        "source_task": row["source_task"],
        "target_task": row["target_task"],
        "layer": row["layer"],
        "view": row["view"],
        "k": row["k"],
        "seed": row["seed"],
        "balance_mode": row["balance_mode"],
        "prediction_split": split,
        "falsification_manifest_id": manifest["manifest_id"],
        "falsification_manifest_sha256": manifest["manifest_sha256"],
        "falsification_task": manifest["task_name"],
    }


def _select_runs(
    results: pd.DataFrame, manifest: dict[str, Any]
) -> list[tuple[pd.Series, str]]:
    selected = results[results["model"].astype(str) == str(manifest["model"])]
    task_name = manifest["task_name"]
    output: list[tuple[pd.Series, str]] = []
    for _, row in selected.iterrows():
        source_task = str(row.get("source_task", ""))
        target_task = str(row.get("target_task", ""))
        if source_task == task_name:
            output.append((row, "source_test"))
        elif target_task == task_name:
            output.append((row, "target_test"))
    return output


def _is_registered_behavior_transfer(
    run: pd.Series,
    manifest: dict[str, Any],
    registry: dict[str, Any] | None,
) -> bool:
    if registry is None:
        return False
    transfer = registry["behavior_transfer"]
    task_name = manifest["task_name"]
    behavior_values = {
        str(example["axes"]["behavior"]["value"]) for example in manifest["examples"]
    }
    return (
        str(run.get("source_task", "")) in transfer["source_values"]
        and str(run.get("target_task", "")) == task_name
        and behavior_values.issubset(set(transfer["heldout_values"]))
    )


def evaluate_manifest(
    results: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    results_dir: Path,
    registry: dict[str, Any] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    test_examples = [
        example
        for example in manifest["examples"]
        if example["protocol_split"] == "test"
    ]
    example_index = {example["example_id"]: example for example in test_examples}
    if not example_index:
        raise ValueError(
            f"Falsification manifest {manifest['manifest_id']} has no test examples"
        )
    slice_rows: list[dict[str, Any]] = []
    pair_prediction_rows: list[dict[str, Any]] = []
    shift_prediction_rows: list[dict[str, Any]] = []
    selected_runs = _select_runs(results, manifest)
    if not selected_runs:
        raise ValueError(
            f"No result runs match falsification manifest {manifest['manifest_id']}"
        )
    for run, split in selected_runs:
        predictions = _load_predictions(str(run["prediction_file"]), results_dir, split)
        predictions["example_id"] = predictions["example_id"].astype(str)
        predictions["question_id"] = predictions["question_id"].astype(str)
        prediction_index = predictions.set_index("example_id", drop=False)
        expected_ids = set(example_index)
        observed_ids = set(prediction_index.index).intersection(expected_ids)
        if observed_ids != expected_ids:
            missing = sorted(expected_ids.difference(observed_ids))
            raise ValueError(
                f"Run {run['run_id']} lacks {len(missing)} falsification examples: {missing[:5]}"
            )
        common = _run_metadata(run, manifest, split)
        behavior_transfer = _is_registered_behavior_transfer(run, manifest, registry)
        run_shift_rows: list[dict[str, Any]] = []
        for example_id, expected in example_index.items():
            observed = prediction_index.loc[example_id]
            if (
                int(observed["label"]) != expected["label"]
                or str(observed["question_id"]) != expected["group_id"]
            ):
                raise ValueError(
                    f"Run {run['run_id']} disagrees with falsification manifest labels/groups"
                )
            shift_row = {
                **common,
                "example_id": example_id,
                "group_id": expected["group_id"],
                "question_id": expected["group_id"],
                "label": int(observed["label"]),
                "score": float(observed["score"]),
                "predicted_positive": bool(observed["predicted_positive"]),
            }
            for axis in SHIFT_AXES:
                shift_row[f"{axis}_value"] = str(expected["axes"][axis]["value"])
                shift_row[f"{axis}_role"] = (
                    "heldout"
                    if axis == "behavior" and behavior_transfer
                    else str(expected["axes"][axis]["role"])
                )
            run_shift_rows.append(shift_row)
        shift_prediction_rows.extend(run_shift_rows)
        run_shift_frame = pd.DataFrame(run_shift_rows)
        for axis in SHIFT_AXES:
            values = sorted(
                {
                    (str(value), str(role))
                    for value, role in zip(
                        run_shift_frame[f"{axis}_value"],
                        run_shift_frame[f"{axis}_role"],
                    )
                }
            )
            for value, role in values:
                subset = run_shift_frame[
                    (run_shift_frame[f"{axis}_value"].astype(str) == value)
                    & (run_shift_frame[f"{axis}_role"].astype(str) == role)
                ]
                slice_rows.append(
                    {
                        **common,
                        "slice_type": "shift",
                        "axis": axis,
                        "value": value,
                        "role": role,
                        **_slice_metrics(subset),
                    }
                )

        pairs = manifest["hard_negative_pairs"]
        if pairs:
            positive_scores: list[float] = []
            negative_scores: list[float] = []
            positive_predictions: list[bool] = []
            negative_predictions: list[bool] = []
            for pair in pairs:
                positive = prediction_index.loc[pair["positive_example_id"]]
                negative = prediction_index.loc[pair["negative_example_id"]]
                if int(positive["label"]) != 1 or int(negative["label"]) != 0:
                    raise ValueError(
                        f"Run {run['run_id']} reverses a hard-negative pair"
                    )
                positive_score = float(positive["score"])
                negative_score = float(negative["score"])
                positive_prediction = bool(positive["predicted_positive"])
                negative_prediction = bool(negative["predicted_positive"])
                positive_scores.append(positive_score)
                negative_scores.append(negative_score)
                positive_predictions.append(positive_prediction)
                negative_predictions.append(negative_prediction)
                pair_prediction_rows.append(
                    {
                        **common,
                        "pair_id": pair["pair_id"],
                        "group_id": pair["group_id"],
                        "scenario_id": pair["scenario_id"],
                        "positive_example_id": pair["positive_example_id"],
                        "negative_example_id": pair["negative_example_id"],
                        "positive_score": positive_score,
                        "negative_score": negative_score,
                        "score_margin": positive_score - negative_score,
                        "positive_detected": positive_prediction,
                        "hard_negative_false_positive": negative_prediction,
                    }
                )
            positive_array = np.asarray(positive_scores)
            negative_array = np.asarray(negative_scores)
            ordering = np.where(
                positive_array > negative_array,
                1.0,
                np.where(positive_array == negative_array, 0.5, 0.0),
            )
            slice_rows.append(
                {
                    **common,
                    "slice_type": "matched_hard_negative",
                    "axis": "hard_negative",
                    "value": "exact_trigger_prompt",
                    "role": "heldout",
                    "n_examples": int(2 * len(pairs)),
                    "n_groups": int(len({pair["group_id"] for pair in pairs})),
                    "n_positive": int(len(pairs)),
                    "n_negative": int(len(pairs)),
                    "auroc": compute_auroc(
                        np.concatenate([np.ones(len(pairs)), np.zeros(len(pairs))]),
                        np.concatenate([positive_array, negative_array]),
                    ),
                    "auprc": compute_auprc(
                        np.concatenate([np.ones(len(pairs)), np.zeros(len(pairs))]),
                        np.concatenate([positive_array, negative_array]),
                    ),
                    "tpr_at_frozen_threshold": float(np.mean(positive_predictions)),
                    "fpr_at_frozen_threshold": float(np.mean(negative_predictions)),
                    "mean_positive_score": float(np.mean(positive_array)),
                    "mean_negative_score": float(np.mean(negative_array)),
                    "pairwise_order_accuracy": float(np.mean(ordering)),
                    "mean_pairwise_score_margin": float(
                        np.mean(positive_array - negative_array)
                    ),
                }
            )
    return slice_rows, pair_prediction_rows, shift_prediction_rows


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen monitor predictions on registered shift slices"
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = collect_results(str(results_dir))
    if results.empty:
        raise ValueError("No monitor results found")
    if "status" in results.columns:
        results = results[results["status"] == "ok"]
    required_result_columns = {
        "run_id",
        "probe",
        "model",
        "source_task",
        "target_task",
        "layer",
        "view",
        "k",
        "seed",
        "balance_mode",
        "prediction_file",
    }
    missing = sorted(required_result_columns.difference(results.columns))
    if missing:
        raise ValueError(f"Monitor results lack fields {missing}")
    registry, registry_sha256 = load_falsification_registry(Path(args.registry))
    archive_dir = results_dir / "falsification_manifests"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_registry = archive_dir / "falsification_registry.yaml"
    shutil.copy2(Path(args.registry), archived_registry)
    slice_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    shift_prediction_rows: list[dict[str, Any]] = []
    seen_manifests: set[str] = set()
    artifact_index: list[dict[str, str]] = []
    for path_value in args.manifests:
        source_manifest_path = Path(path_value)
        manifest = load_falsification_evaluation_manifest(
            source_manifest_path,
            registry=registry,
            registry_sha256=registry_sha256,
            check_source=True,
        )
        if manifest["manifest_id"] in seen_manifests:
            raise ValueError(
                f"Duplicate falsification manifest {manifest['manifest_id']}"
            )
        seen_manifests.add(manifest["manifest_id"])
        archived_name = f"{manifest['manifest_id']}.json"
        archived_path = archive_dir / archived_name
        shutil.copy2(source_manifest_path, archived_path)
        artifact_index.append(
            {
                "manifest_id": manifest["manifest_id"],
                "manifest_sha256": manifest["manifest_sha256"],
                "model": manifest["model"],
                "task_name": manifest["task_name"],
                "archived_file": archived_name,
            }
        )
        slices, pairs, shift_predictions = evaluate_manifest(
            results,
            manifest,
            results_dir=results_dir,
            registry=registry,
        )
        slice_rows.extend(slices)
        pair_rows.extend(pairs)
        shift_prediction_rows.extend(shift_predictions)
    if not slice_rows:
        raise ValueError("No falsification slice results were produced")
    slice_path = results_dir / "falsification_slices.csv"
    temporary_csv = slice_path.with_suffix(".csv.tmp")
    pd.DataFrame(slice_rows).to_csv(temporary_csv, index=False)
    temporary_csv.replace(slice_path)
    pair_path = results_dir / "falsification_pair_predictions.jsonl"
    _atomic_write_jsonl(pair_path, pair_rows)
    shift_prediction_path = results_dir / "falsification_shift_predictions.jsonl"
    _atomic_write_jsonl(shift_prediction_path, shift_prediction_rows)
    _atomic_write_json(
        archive_dir / "artifact_index.json",
        {
            "schema_version": "frontier-falsification-artifact-index-v1",
            "registry_id": registry["registry_id"],
            "registry_sha256": registry_sha256,
            "registry_file": archived_registry.name,
            "manifests": sorted(
                artifact_index, key=lambda row: (row["model"], row["task_name"])
            ),
            "evidence": {
                "slice_summary": {
                    "file": slice_path.name,
                    "sha256": file_sha256(slice_path),
                },
                "shift_predictions": {
                    "file": shift_prediction_path.name,
                    "sha256": file_sha256(shift_prediction_path),
                },
                "pair_predictions": {
                    "file": pair_path.name,
                    "sha256": file_sha256(pair_path),
                },
            },
        },
    )
    print(f"saved {len(slice_rows)} falsification slice rows to {slice_path}")
    print(
        f"saved {len(shift_prediction_rows)} shift predictions to {shift_prediction_path}"
    )
    print(f"saved {len(pair_rows)} matched-pair predictions to {pair_path}")


if __name__ == "__main__":
    main()
