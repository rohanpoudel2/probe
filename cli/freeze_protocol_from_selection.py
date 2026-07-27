from __future__ import annotations

import argparse
import copy
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cli.common import load_yaml
from data.falsification import (
    SHIFT_AXES,
    load_falsification_registry,
    validate_falsification_comparisons,
)
from evaluation.aggregation import collect_results
from evaluation.task_selection import select_primary_source_systems


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configured_pairs(config: dict[str, Any]) -> list[tuple[str, str]]:
    pairs = {
        (str(row["source_task"]), str(row["target_task"]))
        for key in ("task_pairs", "calibration_pairs", "transfer_pairs")
        for row in config.get(key, [])
    }
    if not pairs:
        raise ValueError("Protocol config contains no task pairs")
    return sorted(pairs)


def _identity(row: pd.Series, *, include_balance: bool) -> dict[str, Any]:
    layer = float(row["layer"])
    identity: dict[str, Any] = {
        "probe": str(row["probe"]),
        "layer": int(layer) if layer.is_integer() else layer,
        "view": str(row["view"]),
    }
    if include_balance:
        identity["balance_mode"] = str(row["balance_mode"])
    return identity


def _selection_lookup(
    selected: pd.DataFrame,
) -> dict[tuple[str, str, str], pd.Series]:
    required = {
        "model",
        "source_task",
        "access_regime",
        "probe",
        "balance_mode",
        "layer",
        "view",
    }
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"Source-selection file lacks columns {missing}")
    lookup: dict[tuple[str, str, str], pd.Series] = {}
    for _, row in selected.iterrows():
        key = (
            str(row["model"]),
            str(row["source_task"]),
            str(row["access_regime"]),
        )
        if key in lookup:
            raise ValueError(f"Duplicate frozen source selection for {key}")
        lookup[key] = row
    return lookup


def _canonical_selection(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "source_task",
        "access_regime",
        "probe",
        "balance_mode",
        "layer",
        "view",
        "selection_k",
        "selection_metric",
        "selection_rule",
    ]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Source-selection file lacks columns {missing}")
    canonical = frame[columns].copy()
    canonical["layer"] = pd.to_numeric(canonical["layer"], errors="raise")
    canonical["selection_k"] = pd.to_numeric(
        canonical["selection_k"], errors="raise"
    ).astype(int)
    for column in set(columns).difference({"layer", "selection_k"}):
        canonical[column] = canonical[column].astype(str)
    return canonical.sort_values(
        ["model", "source_task", "access_regime"], kind="mergesort"
    ).reset_index(drop=True)


def validate_selection_evidence(
    config: dict[str, Any],
    results_dir: Path,
    selected: pd.DataFrame,
    *,
    selection_k: int,
) -> None:
    """Verify that the freeze input came from a source-only execution."""

    if config.get("execution_mode") != "selection":
        raise ValueError(
            "Protocol freezing requires a base config with execution_mode=selection"
        )
    raw = collect_results(results_dir)
    if raw.empty:
        raise ValueError(f"No selection run summaries found in {results_dir}")
    required = {"run_id", "status", "execution_mode"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Selection run summaries lack columns {missing}")
    if not (raw["status"].astype(str) == "ok").all():
        raise ValueError("Protocol freezing refuses failed or partial selection runs")
    if set(raw["execution_mode"].astype(str)) != {"selection"}:
        raise ValueError(
            "Protocol freezing found run summaries not marked execution_mode=selection"
        )
    if not raw["run_id"].astype(str).str.endswith("__selection").all():
        raise ValueError("Selection run IDs lack the immutable __selection suffix")

    for prefix in ("test_", "transfer_"):
        for column in (
            name for name in raw.columns if str(name).startswith(prefix)
        ):
            values = pd.to_numeric(raw[column], errors="coerce")
            if values.notna().any():
                raise ValueError(
                    f"Selection evidence contains behavior-test values in {column}"
                )

    summary_path = results_dir / "task_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}; aggregate selection runs before freezing"
        )
    summary = pd.read_csv(summary_path)
    seed_count_column = "eval_tpr_at_reference_alert_budget_n_seeds"
    if seed_count_column not in summary:
        raise ValueError(
            f"Selection summary lacks completeness field {seed_count_column}"
        )
    expected_seeds = int(config["seeds"])
    seed_counts = pd.to_numeric(
        summary[seed_count_column], errors="raise"
    ).astype(int)
    if not (seed_counts == expected_seeds).all():
        raise ValueError(
            "Selection summary contains incomplete few-shot seed groups"
        )
    at_selection_k = summary[
        pd.to_numeric(summary["k"], errors="raise").astype(int)
        == int(selection_k)
    ].copy()
    configured_probes = {
        value.strip()
        for value in str(config["probes"]).split(",")
        if value.strip()
    }
    if config.get("run_black_box_baselines"):
        configured_probes.update(
            str(value) for value in config.get("black_box_baselines", [])
        )
    for model in config["models"]:
        model_name = str(model["name"])
        for source_task in {
            source for source, _ in _configured_pairs(config)
        }:
            rows = at_selection_k[
                (at_selection_k["model"].astype(str) == model_name)
                & (at_selection_k["source_task"].astype(str) == source_task)
            ]
            observed_probes = set(rows["probe"].astype(str))
            missing_probes = configured_probes.difference(observed_probes)
            if missing_probes:
                raise ValueError(
                    f"Selection evidence lacks registered probes for "
                    f"{model_name}/{source_task}: {sorted(missing_probes)}"
                )
            white_rows = rows[rows["probe"].astype(str).str.startswith("P")]
            white_identity_sets = {
                probe: {
                    (str(row.layer), str(row.view))
                    for row in probe_rows.itertuples(index=False)
                }
                for probe, probe_rows in white_rows.groupby("probe")
            }
            if len({frozenset(value) for value in white_identity_sets.values()}) > 1:
                raise ValueError(
                    f"Selection evidence has incomplete layer/view coverage for "
                    f"{model_name}/{source_task}"
                )

            expected_black_views = {
                "B1_text_tfidf": set(config.get("text_embedding_views", [])),
                "B2_text_embedding_logistic": set(
                    config.get("text_embedding_views", [])
                ),
                "B3_llm_judge_zero_shot": set(config.get("llm_judge_views", [])),
                "B3_llm_judge_few_shot": set(config.get("llm_judge_views", [])),
                "B4_output_confidence_logistic": {"generation_confidence"},
            }
            for probe, expected_views in expected_black_views.items():
                if probe not in configured_probes:
                    continue
                observed_views = set(
                    rows.loc[
                        rows["probe"].astype(str) == probe,
                        "view",
                    ].astype(str)
                )
                if observed_views != expected_views:
                    raise ValueError(
                        f"Selection evidence has incomplete views for "
                        f"{model_name}/{source_task}/{probe}: "
                        f"expected={sorted(expected_views)}, "
                        f"observed={sorted(observed_views)}"
                    )
    recomputed = select_primary_source_systems(
        summary,
        selection_k=int(selection_k),
    )
    frozen_input = _canonical_selection(selected)
    expected = _canonical_selection(recomputed)
    if not frozen_input.equals(expected):
        raise ValueError(
            "task_primary_source_systems.csv does not match a fresh deterministic "
            "selection from task_summary.csv"
        )

    expected_keys = {
        (str(model["name"]), source_task, access_regime)
        for model in config["models"]
        for source_task in {source for source, _ in _configured_pairs(config)}
        for access_regime in ("white_box", "black_box")
    }
    observed_keys = {
        (row.model, row.source_task, row.access_regime)
        for row in frozen_input.itertuples(index=False)
    }
    if observed_keys != expected_keys:
        raise ValueError(
            "Source selection has incomplete model/task/access coverage: "
            f"missing={sorted(expected_keys - observed_keys)[:5]}, "
            f"extra={sorted(observed_keys - expected_keys)[:5]}"
        )


def build_primary_comparisons(
    config: dict[str, Any],
    selected: pd.DataFrame,
) -> dict[str, Any]:
    lookup = _selection_lookup(selected)
    k_values = sorted(
        {
            int(value.strip())
            for value in str(config["k_values"]).split(",")
            if value.strip()
        }
    )
    comparisons: list[dict[str, Any]] = []
    for model in config["models"]:
        model_name = str(model["name"])
        for source_task, target_task in _configured_pairs(config):
            white = lookup.get((model_name, source_task, "white_box"))
            black = lookup.get((model_name, source_task, "black_box"))
            if white is None or black is None:
                raise ValueError(
                    f"Missing white/black source selection for {model_name}/{source_task}"
                )
            for k in k_values:
                comparisons.append(
                    {
                        "comparison_id": (
                            f"primary-{_slug(model_name)}-{_slug(source_task)}-"
                            f"to-{_slug(target_task)}-k{k}"
                        ),
                        "description": (
                            "Source-selected activation-monitor TPR minus the "
                            "source-selected transcript-only monitor TPR."
                        ),
                        "comparison_role": "primary_white_box_gain",
                        "common_filters": {
                            "model": model_name,
                            "source_task": source_task,
                            "target_task": target_task,
                            "k": k,
                        },
                        "system_a": _identity(white, include_balance=True),
                        "system_b": _identity(black, include_balance=True),
                        "split": "target_test",
                        "metric": "tpr",
                    }
                )
    return {
        "schema_version": "frontier-primary-comparisons-v1",
        "multiplicity_control": "holm_global",
        "comparisons": comparisons,
    }


def build_falsification_comparisons(
    config: dict[str, Any],
    selected: pd.DataFrame,
    registry: dict[str, Any],
    *,
    selection_k: int,
) -> dict[str, Any]:
    lookup = _selection_lookup(selected)
    comparisons: list[dict[str, Any]] = []
    behavior_transfer = registry["behavior_transfer"]
    for model in config["models"]:
        model_name = str(model["name"])
        for source_task, target_task in _configured_pairs(config):
            if target_task not in registry["tasks"]:
                continue
            white = lookup.get((model_name, source_task, "white_box"))
            black = lookup.get((model_name, source_task, "black_box"))
            if white is None or black is None:
                raise ValueError(
                    f"Missing white/black source selection for {model_name}/{source_task}"
                )
            common = {
                "model": model_name,
                "source_task": source_task,
                "target_task": target_task,
                "k": int(selection_k),
            }
            systems = {
                "system_a": _identity(white, include_balance=True),
                "system_b": _identity(black, include_balance=True),
            }
            task_values = registry["tasks"][target_task]["values"]
            slices: list[tuple[str, str]] = []
            if (
                source_task in behavior_transfer["source_values"]
                and target_task in behavior_transfer["heldout_values"]
            ):
                slices.append(("behavior", target_task))
            for axis in SHIFT_AXES:
                if axis == "behavior":
                    continue
                slices.extend(
                    (axis, str(value))
                    for value in task_values[axis]["heldout"]
                )
            for axis, value in slices:
                comparisons.append(
                    {
                        "comparison_id": (
                            f"shift-{_slug(model_name)}-{_slug(source_task)}-to-"
                            f"{_slug(target_task)}-{_slug(axis)}-{_slug(value)}"
                        ),
                        "description": (
                            "Activation-monitor TPR minus transcript-only monitor "
                            f"TPR on the registered held-out {axis}={value} slice."
                        ),
                        "task_name": target_task,
                        "slice": {
                            "type": "shift",
                            "axis": axis,
                            "value": value,
                            "role": "heldout",
                        },
                        "common_filters": common,
                        **systems,
                        "metric": "tpr",
                    }
                )
            if registry["tasks"][target_task]["hard_negative"]["enabled"]:
                for metric in ("hard_negative_fpr", "pairwise_order_accuracy"):
                    comparisons.append(
                        {
                            "comparison_id": (
                                f"hard-{_slug(model_name)}-{_slug(source_task)}-to-"
                                f"{_slug(target_task)}-{_slug(metric)}"
                            ),
                            "description": (
                                "Activation-monitor minus transcript-only monitor "
                                f"on exact-prompt {metric}."
                            ),
                            "task_name": target_task,
                            "slice": {"type": "matched_hard_negative"},
                            "common_filters": common,
                            **systems,
                            "metric": metric,
                        }
                    )
    output = {
        "schema_version": "frontier-falsification-comparisons-v1",
        "multiplicity_control": "holm_global",
        "comparisons": comparisons,
    }
    validate_falsification_comparisons(output, registry=registry)
    return output


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == rendered:
            return
        raise FileExistsError(f"Refusing to replace frozen protocol artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _yaml_sha256(payload: dict[str, Any]) -> str:
    rendered = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze confirmatory protocol files from source-only selections"
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--selection-results-dir", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--comparisons-output", required=True)
    parser.add_argument("--falsification-comparisons-output", required=True)
    parser.add_argument(
        "--selection-k",
        type=int,
        default=None,
        help="Optional assertion; it must match the base manifest selection_k.",
    )
    parser.add_argument(
        "--confirmatory-results-dir", default="results/frontier_main"
    )
    args = parser.parse_args()

    base_path = Path(args.base_config)
    base = load_yaml(base_path)
    registered_selection_k = int(base["selection_k"])
    if (
        args.selection_k is not None
        and int(args.selection_k) != registered_selection_k
    ):
        raise ValueError(
            f"--selection-k={args.selection_k} does not match the registered "
            f"selection_k={registered_selection_k}"
        )
    selection_path = (
        Path(args.selection_results_dir) / "task_primary_source_systems.csv"
    )
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Missing {selection_path}; complete a selection-only run first"
        )
    selected = pd.read_csv(selection_path)
    validate_selection_evidence(
        base,
        Path(args.selection_results_dir),
        selected,
        selection_k=registered_selection_k,
    )
    registry_path = Path(base["falsification_registry"])
    registry, registry_sha256 = load_falsification_registry(registry_path)
    primary = build_primary_comparisons(base, selected)
    falsification = build_falsification_comparisons(
        base,
        selected,
        registry,
        selection_k=registered_selection_k,
    )

    primary_path = Path(args.comparisons_output)
    falsification_path = Path(args.falsification_comparisons_output)
    selection_provenance = {
        "selection_file": str(selection_path),
        "selection_file_sha256": _sha256(selection_path),
        "base_config_sha256": _sha256(base_path),
        "selection_k": registered_selection_k,
        "registry_sha256": registry_sha256,
    }
    primary["selection_provenance"] = copy.deepcopy(selection_provenance)
    falsification["selection_provenance"] = copy.deepcopy(selection_provenance)
    frozen = copy.deepcopy(base)
    frozen["protocol_stage"] = "frozen"
    frozen["execution_mode"] = "confirmatory"
    frozen["bootstrap_samples"] = max(int(frozen.get("bootstrap_samples", 0)), 5000)
    frozen["results_dir"] = args.confirmatory_results_dir
    frozen["comparisons_file"] = str(primary_path)
    frozen["falsification_comparisons_file"] = str(falsification_path)
    frozen["selection_provenance"] = copy.deepcopy(selection_provenance)
    frozen["registered_artifact_sha256"] = {
        "comparisons_file": _yaml_sha256(primary),
        "falsification_comparisons_file": _yaml_sha256(falsification),
    }

    _atomic_yaml(primary_path, primary)
    _atomic_yaml(falsification_path, falsification)
    _atomic_yaml(Path(args.output_config), frozen)
    print(f"saved frozen primary comparisons to {primary_path}")
    print(f"saved frozen falsification comparisons to {falsification_path}")
    print(f"saved frozen confirmatory config to {args.output_config}")


if __name__ == "__main__":
    main()
