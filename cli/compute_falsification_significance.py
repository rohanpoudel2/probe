from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.falsification import (
    FALSIFICATION_COMPARISONS_SCHEMA_VERSION,
    SHIFT_AXES,
    load_falsification_comparisons,
    load_falsification_registry,
)
from evaluation.hierarchical_statistics import (
    hierarchical_paired_mean_difference,
    hierarchical_paired_rate_difference,
    holm_adjust,
)


SHIFT_REQUIRED_COLUMNS = {
    "run_id",
    "seed",
    "example_id",
    "group_id",
    "label",
    "predicted_positive",
    "falsification_manifest_id",
    "falsification_manifest_sha256",
    "falsification_task",
} | {f"{axis}_{suffix}" for axis in SHIFT_AXES for suffix in ("value", "role")}
PAIR_REQUIRED_COLUMNS = {
    "run_id",
    "seed",
    "pair_id",
    "group_id",
    "scenario_id",
    "positive_example_id",
    "negative_example_id",
    "positive_score",
    "negative_score",
    "score_margin",
    "positive_detected",
    "hard_negative_false_positive",
    "falsification_manifest_id",
    "falsification_manifest_sha256",
    "falsification_task",
}


def _read_jsonl(path: Path, *, required: set[str], allow_empty: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing falsification prediction artifact: {path}")
    if path.stat().st_size == 0:
        if allow_empty:
            return pd.DataFrame(columns=sorted(required))
        raise ValueError(f"Falsification prediction artifact is empty: {path}")
    frame = pd.read_json(path, lines=True)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Falsification prediction artifact {path} lacks {missing}")
    return frame


def _apply_filters(
    frame: pd.DataFrame,
    filters: dict[str, Any],
    *,
    comparison_id: str,
) -> pd.DataFrame:
    selected = frame
    for key, value in filters.items():
        if key not in selected.columns:
            raise ValueError(
                f"Falsification comparison {comparison_id} filters on absent column {key!r}"
            )
        selected = selected[selected[key].astype(str) == str(value)]
    if selected.empty:
        raise ValueError(
            f"Falsification comparison {comparison_id} selected no predictions"
        )
    runs_per_seed = (
        selected[["seed", "run_id"]]
        .astype(str)
        .drop_duplicates()
        .groupby("seed")["run_id"]
        .nunique()
    )
    if not (runs_per_seed == 1).all():
        raise ValueError(
            f"Falsification comparison {comparison_id} does not select one run per seed"
        )
    return selected.copy()


def _pair_system_rows(
    rows_a: pd.DataFrame,
    rows_b: pd.DataFrame,
    *,
    identity_key: str,
    comparison_id: str,
) -> pd.DataFrame:
    rows_a = rows_a.copy()
    rows_b = rows_b.copy()
    for rows in (rows_a, rows_b):
        rows["seed"] = rows["seed"].astype(str)
        rows[identity_key] = rows[identity_key].astype(str)
    seeds_a = set(rows_a["seed"].astype(str))
    seeds_b = set(rows_b["seed"].astype(str))
    if seeds_a != seeds_b:
        raise ValueError(
            f"Falsification comparison {comparison_id} has different system seeds"
        )
    merge_keys = ["seed", identity_key]
    if (
        rows_a[merge_keys].astype(str).duplicated().any()
        or rows_b[merge_keys].astype(str).duplicated().any()
    ):
        raise ValueError(
            f"Falsification comparison {comparison_id} duplicates paired predictions"
        )
    merged = rows_a.merge(
        rows_b,
        on=merge_keys,
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if len(merged) != len(rows_a) or len(merged) != len(rows_b):
        raise ValueError(
            f"Falsification comparison {comparison_id} systems cover different evidence"
        )
    if not (merged["group_id_a"].astype(str) == merged["group_id_b"].astype(str)).all():
        raise ValueError(
            f"Falsification comparison {comparison_id} disagrees on scenario groups"
        )
    for column in ("falsification_manifest_id", "falsification_manifest_sha256"):
        if not (
            merged[f"{column}_a"].astype(str) == merged[f"{column}_b"].astype(str)
        ).all():
            raise ValueError(
                f"Falsification comparison {comparison_id} mixes evaluation manifests"
            )
    return merged


def _shift_comparison(
    shift_predictions: pd.DataFrame,
    comparison: dict[str, Any],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    comparison_id = comparison["comparison_id"]
    slice_cfg = comparison["slice"]
    axis = slice_cfg["axis"]
    value_column = f"{axis}_value"
    role_column = f"{axis}_role"
    for column in (value_column, role_column):
        if column not in shift_predictions.columns:
            raise ValueError(
                f"Falsification shift predictions lack registered axis column {column}"
            )
    sliced = shift_predictions[
        (shift_predictions["falsification_task"].astype(str) == comparison["task_name"])
        & (shift_predictions[value_column].astype(str) == str(slice_cfg["value"]))
        & (shift_predictions[role_column].astype(str) == str(slice_cfg["role"]))
    ]
    filters_a = {**comparison["common_filters"], **comparison["system_a"]}
    filters_b = {**comparison["common_filters"], **comparison["system_b"]}
    rows_a = _apply_filters(sliced, filters_a, comparison_id=comparison_id)
    rows_b = _apply_filters(sliced, filters_b, comparison_id=comparison_id)
    paired = _pair_system_rows(
        rows_a,
        rows_b,
        identity_key="example_id",
        comparison_id=comparison_id,
    )
    if not (paired["label_a"].astype(int) == paired["label_b"].astype(int)).all():
        raise ValueError(
            f"Falsification comparison {comparison_id} systems disagree on labels"
        )
    return hierarchical_paired_rate_difference(
        paired["label_a"].to_numpy(dtype=np.int64),
        paired["predicted_positive_a"].astype(float).to_numpy(),
        paired["predicted_positive_b"].astype(float).to_numpy(),
        paired["group_id_a"].astype(str).to_numpy(),
        paired["seed"].astype(str).to_numpy(),
        metric=comparison["metric"],
        n_boot=n_boot,
        seed=seed,
    )


def _hard_negative_comparison(
    pair_predictions: pd.DataFrame,
    comparison: dict[str, Any],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    comparison_id = comparison["comparison_id"]
    sliced = pair_predictions[
        pair_predictions["falsification_task"].astype(str) == comparison["task_name"]
    ]
    filters_a = {**comparison["common_filters"], **comparison["system_a"]}
    filters_b = {**comparison["common_filters"], **comparison["system_b"]}
    rows_a = _apply_filters(sliced, filters_a, comparison_id=comparison_id)
    rows_b = _apply_filters(sliced, filters_b, comparison_id=comparison_id)
    paired = _pair_system_rows(
        rows_a,
        rows_b,
        identity_key="pair_id",
        comparison_id=comparison_id,
    )
    for column in ("positive_example_id", "negative_example_id", "scenario_id"):
        if (
            f"{column}_a" in paired.columns
            and not (
                paired[f"{column}_a"].astype(str) == paired[f"{column}_b"].astype(str)
            ).all()
        ):
            raise ValueError(
                f"Falsification comparison {comparison_id} disagrees on paired evidence"
            )
    metric = comparison["metric"]
    if metric == "hard_negative_fpr":
        values_a = paired["hard_negative_false_positive_a"].astype(float).to_numpy()
        values_b = paired["hard_negative_false_positive_b"].astype(float).to_numpy()
    elif metric == "paired_positive_tpr":
        values_a = paired["positive_detected_a"].astype(float).to_numpy()
        values_b = paired["positive_detected_b"].astype(float).to_numpy()
    elif metric == "pairwise_order_accuracy":
        margin_a = (
            paired["positive_score_a"].astype(float).to_numpy()
            - paired["negative_score_a"].astype(float).to_numpy()
        )
        margin_b = (
            paired["positive_score_b"].astype(float).to_numpy()
            - paired["negative_score_b"].astype(float).to_numpy()
        )
        values_a = np.where(margin_a > 0, 1.0, np.where(margin_a == 0, 0.5, 0.0))
        values_b = np.where(margin_b > 0, 1.0, np.where(margin_b == 0, 0.5, 0.0))
    elif metric == "mean_pairwise_score_margin":
        values_a = paired["score_margin_a"].astype(float).to_numpy()
        values_b = paired["score_margin_b"].astype(float).to_numpy()
    else:  # validated before execution
        raise ValueError(f"Unsupported hard-negative metric {metric}")
    return hierarchical_paired_mean_difference(
        values_a,
        values_b,
        paired["group_id_a"].astype(str).to_numpy(),
        paired["seed"].astype(str).to_numpy(),
        n_boot=n_boot,
        seed=seed,
    )


def compute_falsification_comparisons(
    shift_predictions: pd.DataFrame,
    pair_predictions: pd.DataFrame,
    comparison_config: dict[str, Any],
    *,
    comparisons_sha256: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    output_rows: list[dict[str, Any]] = []
    for index, comparison in enumerate(comparison_config["comparisons"]):
        slice_cfg = comparison["slice"]
        if slice_cfg["type"] == "shift":
            result = _shift_comparison(
                shift_predictions,
                comparison,
                n_boot=n_boot,
                seed=seed + index,
            )
        else:
            result = _hard_negative_comparison(
                pair_predictions,
                comparison,
                n_boot=n_boot,
                seed=seed + index,
            )
        filters_a = {**comparison["common_filters"], **comparison["system_a"]}
        filters_b = {**comparison["common_filters"], **comparison["system_b"]}
        output_rows.append(
            {
                "comparison_id": comparison["comparison_id"],
                "description": comparison["description"],
                "task_name": comparison["task_name"],
                "slice_type": slice_cfg["type"],
                "axis": slice_cfg.get("axis", "hard_negative"),
                "value": slice_cfg.get("value", "exact_trigger_prompt"),
                "role": slice_cfg.get("role", "heldout"),
                "metric": comparison["metric"],
                "difference_direction": "system_a_minus_system_b",
                "system_a": json.dumps(filters_a, sort_keys=True),
                "system_b": json.dumps(filters_b, sort_keys=True),
                "comparisons_sha256": comparisons_sha256,
                "bootstrap_samples": int(n_boot),
                "bootstrap_seed": int(seed + index),
                **result,
            }
        )
    adjusted = holm_adjust([float(row["p_value"]) for row in output_rows])
    for row, adjusted_p in zip(output_rows, adjusted):
        row["holm_adjusted_p_value"] = adjusted_p
    return pd.DataFrame(output_rows)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _archive_comparisons(
    results_dir: Path,
    comparisons_path: Path,
    *,
    comparisons_sha256: str,
    registry: dict[str, Any],
    registry_sha256: str,
) -> None:
    archive_dir = results_dir / "falsification_manifests"
    index_path = archive_dir / "artifact_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            "Falsification inference requires the evaluator's archived artifact index"
        )
    artifact_index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        artifact_index.get("registry_id") != registry["registry_id"]
        or artifact_index.get("registry_sha256") != registry_sha256
    ):
        raise ValueError("Falsification artifact index uses a different registry")
    evidence = artifact_index.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError(
            "Falsification artifact index lacks hashed prediction evidence"
        )
    for record in evidence.values():
        if not isinstance(record, dict) or not str(record.get("file", "")):
            raise ValueError(
                "Falsification artifact index has invalid evidence records"
            )
        evidence_path = results_dir / str(record["file"])
        if not evidence_path.exists() or hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest() != record.get("sha256"):
            raise ValueError(
                f"Falsification prediction evidence failed its archived hash: {evidence_path}"
            )
    archived_name = "falsification_comparisons.yaml"
    archived_path = archive_dir / archived_name
    if archived_path.exists():
        archived_hash = hashlib.sha256(archived_path.read_bytes()).hexdigest()
        if archived_hash != comparisons_sha256:
            raise ValueError(
                "Refusing to replace the archived pre-registered falsification comparisons"
            )
    elif comparisons_path.resolve() != archived_path.resolve():
        shutil.copy2(comparisons_path, archived_path)
    artifact_index["comparisons"] = {
        "schema_version": FALSIFICATION_COMPARISONS_SCHEMA_VERSION,
        "file": archived_name,
        "sha256": comparisons_sha256,
        "multiplicity_control": "holm_global",
    }
    _atomic_write_json(index_path, artifact_index)


def _record_significance_artifact(
    results_dir: Path,
    output_path: Path,
    *,
    comparisons_sha256: str,
) -> None:
    index_path = results_dir / "falsification_manifests" / "artifact_index.json"
    artifact_index = json.loads(index_path.read_text(encoding="utf-8"))
    if (artifact_index.get("comparisons") or {}).get("sha256") != comparisons_sha256:
        raise ValueError(
            "Falsification significance uses an unarchived comparison file"
        )
    artifact_index["significance"] = {
        "file": output_path.name,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "comparisons_sha256": comparisons_sha256,
    }
    _atomic_write_json(index_path, artifact_index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pre-registered paired inference on falsification slices"
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--comparisons", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    registry, registry_sha256 = load_falsification_registry(Path(args.registry))
    comparisons_path = Path(args.comparisons)
    comparison_config, comparisons_sha256 = load_falsification_comparisons(
        comparisons_path,
        registry=registry,
    )
    shift_predictions = _read_jsonl(
        results_dir / "falsification_shift_predictions.jsonl",
        required=SHIFT_REQUIRED_COLUMNS,
        allow_empty=False,
    )
    pair_predictions = _read_jsonl(
        results_dir / "falsification_pair_predictions.jsonl",
        required=PAIR_REQUIRED_COLUMNS,
        allow_empty=True,
    )
    _archive_comparisons(
        results_dir,
        comparisons_path,
        comparisons_sha256=comparisons_sha256,
        registry=registry,
        registry_sha256=registry_sha256,
    )
    output = compute_falsification_comparisons(
        shift_predictions,
        pair_predictions,
        comparison_config,
        comparisons_sha256=comparisons_sha256,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )
    output_path = results_dir / "falsification_significance.csv"
    temporary = output_path.with_suffix(".csv.tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(output_path)
    _record_significance_artifact(
        results_dir,
        output_path,
        comparisons_sha256=comparisons_sha256,
    )
    print(
        f"saved {len(output)} pre-registered falsification comparisons to {output_path}"
    )


if __name__ == "__main__":
    main()
