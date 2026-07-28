from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from data.rollout_schema import content_hash


FALSIFICATION_REGISTRY_SCHEMA_VERSION = "frontier-falsification-registry-v1"
FALSIFICATION_EXAMPLE_SCHEMA_VERSION = "frontier-falsification-example-v1"
FALSIFICATION_EVALUATION_SCHEMA_VERSION = "frontier-falsification-evaluation-v1"
FALSIFICATION_COMPARISONS_SCHEMA_VERSION = "frontier-falsification-comparisons-v1"
SHIFT_AXES = ("behavior", "domain", "template", "presentation", "obfuscation")
TRANSFORMED_AXES = {"presentation", "obfuscation"}
SHIFT_COMPARISON_METRICS = {"tpr", "fpr", "positive_rate"}
HARD_NEGATIVE_COMPARISON_METRICS = {
    "hard_negative_fpr",
    "paired_positive_tpr",
    "pairwise_order_accuracy",
    "mean_pairwise_score_margin",
}
FALSIFICATION_RUN_SELECTOR_KEYS = {
    "model",
    "source_task",
    "target_task",
    "probe",
    "layer",
    "view",
    "k",
    "balance_mode",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_falsification_registry(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing falsification registry: {path}")
    registry = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    validate_falsification_registry(registry)
    return registry, file_sha256(path)


def load_falsification_comparisons(
    path: Path, *, registry: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing falsification comparisons file: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    validate_falsification_comparisons(config, registry=registry)
    return config, file_sha256(path)


def _validate_scalar_filters(filters: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(filters, dict):
        raise ValueError(f"{field} must be an object")
    for key, value in filters.items():
        if not str(key).strip() or value is None or isinstance(value, (dict, list)):
            raise ValueError(f"{field} must contain named scalar selectors")
    if "seed" in filters:
        raise ValueError(f"{field} cannot select a seed; inference pairs every seed")
    return filters


def validate_falsification_comparisons(
    config: dict[str, Any], *, registry: dict[str, Any]
) -> None:
    if config.get("schema_version") != FALSIFICATION_COMPARISONS_SCHEMA_VERSION:
        raise ValueError("Unsupported falsification-comparisons schema")
    if config.get("multiplicity_control") != "holm_global":
        raise ValueError("Falsification comparisons must use global Holm correction")
    comparisons = config.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("Falsification comparisons file must define comparisons")

    comparison_ids: set[str] = set()
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, dict):
            raise ValueError(f"Falsification comparison {index} must be an object")
        comparison_id = str(comparison.get("comparison_id", "")).strip()
        if not comparison_id or comparison_id in comparison_ids:
            raise ValueError(
                "Falsification comparison IDs must be unique and non-empty"
            )
        comparison_ids.add(comparison_id)
        if "REPLACE" in str(comparison):
            raise ValueError(
                f"Falsification comparison {comparison_id} contains placeholder values"
            )
        if not str(comparison.get("description", "")).strip():
            raise ValueError(
                f"Falsification comparison {comparison_id} lacks description"
            )
        task_name = str(comparison.get("task_name", ""))
        if task_name not in registry["tasks"]:
            raise ValueError(
                f"Falsification comparison {comparison_id} has unknown task {task_name!r}"
            )

        common = _validate_scalar_filters(
            comparison.get("common_filters"),
            field=f"{comparison_id}.common_filters",
        )
        system_a = _validate_scalar_filters(
            comparison.get("system_a"), field=f"{comparison_id}.system_a"
        )
        system_b = _validate_scalar_filters(
            comparison.get("system_b"), field=f"{comparison_id}.system_b"
        )
        selectors_a = {**common, **system_a}
        selectors_b = {**common, **system_b}
        for name, selectors in (("A", selectors_a), ("B", selectors_b)):
            missing = FALSIFICATION_RUN_SELECTOR_KEYS.difference(selectors)
            if missing:
                raise ValueError(
                    f"Falsification comparison {comparison_id} system {name} lacks "
                    f"exact selectors {sorted(missing)}"
                )
        if selectors_a == selectors_b:
            raise ValueError(
                f"Falsification comparison {comparison_id} selects the same system twice"
            )
        if task_name not in {
            str(selectors_a["source_task"]),
            str(selectors_a["target_task"]),
        } or task_name not in {
            str(selectors_b["source_task"]),
            str(selectors_b["target_task"]),
        }:
            raise ValueError(
                f"Falsification comparison {comparison_id} task is absent from a selected run"
            )

        slice_cfg = comparison.get("slice")
        if not isinstance(slice_cfg, dict):
            raise ValueError(f"Falsification comparison {comparison_id} lacks slice")
        slice_type = str(slice_cfg.get("type", ""))
        metric = str(comparison.get("metric", ""))
        if slice_type == "shift":
            if set(slice_cfg) != {"type", "axis", "value", "role"}:
                raise ValueError(
                    f"Falsification comparison {comparison_id} shift slice has incomplete or extra fields"
                )
            axis = str(slice_cfg.get("axis", ""))
            value = str(slice_cfg.get("value", ""))
            role = str(slice_cfg.get("role", ""))
            if axis not in SHIFT_AXES or role not in {"source", "heldout"} or not value:
                raise ValueError(
                    f"Falsification comparison {comparison_id} has invalid shift slice"
                )
            if metric not in SHIFT_COMPARISON_METRICS:
                raise ValueError(
                    f"Falsification comparison {comparison_id} has invalid shift metric"
                )
            task_values = registry["tasks"][task_name]["values"][axis]
            if axis == "behavior" and role == "heldout":
                transfer = registry["behavior_transfer"]
                if (
                    value not in transfer["heldout_values"]
                    or value not in task_values["source"]
                    or str(selectors_a["source_task"]) not in transfer["source_values"]
                    or str(selectors_b["source_task"]) not in transfer["source_values"]
                    or str(selectors_a["target_task"]) != task_name
                    or str(selectors_b["target_task"]) != task_name
                ):
                    raise ValueError(
                        f"Falsification comparison {comparison_id} is not a registered behavior transfer"
                    )
            elif value not in task_values[role]:
                raise ValueError(
                    f"Falsification comparison {comparison_id} references an unregistered slice"
                )
        elif slice_type == "matched_hard_negative":
            if set(slice_cfg) != {"type"}:
                raise ValueError(
                    f"Falsification comparison {comparison_id} hard-negative slice has extra fields"
                )
            if not registry["tasks"][task_name]["hard_negative"]["enabled"]:
                raise ValueError(
                    f"Falsification comparison {comparison_id} uses disabled hard negatives"
                )
            if metric not in HARD_NEGATIVE_COMPARISON_METRICS:
                raise ValueError(
                    f"Falsification comparison {comparison_id} has invalid hard-negative metric"
                )
        else:
            raise ValueError(
                f"Falsification comparison {comparison_id} has unknown slice type"
            )


def _unique_strings(value: Any, *, field: str, allow_empty: bool) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{field} must be {qualifier}")
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique non-empty strings")
    return normalized


def validate_falsification_registry(registry: dict[str, Any]) -> None:
    if registry.get("schema_version") != FALSIFICATION_REGISTRY_SCHEMA_VERSION:
        raise ValueError("Unsupported falsification-registry schema")
    if not str(registry.get("registry_id", "")).strip():
        raise ValueError("Falsification registry requires registry_id")
    axes = _unique_strings(registry.get("axes"), field="axes", allow_empty=False)
    if tuple(axes) != SHIFT_AXES:
        raise ValueError(f"Falsification axes must be registered in order {SHIFT_AXES}")
    transformed_axes = set(
        _unique_strings(
            registry.get("transformed_axes"),
            field="transformed_axes",
            allow_empty=False,
        )
    )
    if transformed_axes != TRANSFORMED_AXES:
        raise ValueError(f"transformed_axes must be {sorted(TRANSFORMED_AXES)}")

    shift_protocol = registry.get("shift_protocol")
    if (
        not isinstance(shift_protocol, dict)
        or shift_protocol.get("require_heldout_test_only") is not True
    ):
        raise ValueError("Falsification registry must isolate held-out shifts to test")
    if int(shift_protocol.get("min_independent_groups_per_axis_final", 0)) < 1:
        raise ValueError("Final shift-group minimum must be positive")

    validation = registry.get("transformation_validation")
    if (
        not isinstance(validation, dict)
        or validation.get("protocol") != "deterministic-invariant-transform-v1"
    ):
        raise ValueError(
            "Falsification registry requires deterministic transformation validation"
        )
    for key in (
        "require_executable_inverse",
        "require_verbatim_payload_binding",
        "require_unchanged_answer_metadata",
    ):
        if validation.get(key) is not True:
            raise ValueError(f"transformation_validation.{key} must be true")
    generators = validation.get("allowed_generators")
    if (
        not isinstance(generators, dict)
        or set(generators) != TRANSFORMED_AXES
        or any(not isinstance(generators[axis], dict) or not generators[axis] for axis in TRANSFORMED_AXES)
    ):
        raise ValueError(
            "transformation_validation must register generators for both transformed axes"
        )

    hard_protocol = registry.get("hard_negative_protocol")
    if not isinstance(hard_protocol, dict):
        raise ValueError("Falsification registry requires hard_negative_protocol")
    if hard_protocol.get("match_type") != "exact_trigger_prompt":
        raise ValueError("Hard negatives must use exact_trigger_prompt matching")
    for key in (
        "require_same_scenario",
        "require_same_shift_signature",
        "require_test_split",
    ):
        if hard_protocol.get(key) is not True:
            raise ValueError(f"hard_negative_protocol.{key} must be true")
    if (
        int(hard_protocol.get("min_pairs_selection", 0)) < 1
        or int(hard_protocol.get("min_independent_groups_final", 0)) < 1
    ):
        raise ValueError("Hard-negative minimums must be positive")

    tasks = registry.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Falsification registry requires task definitions")
    for task_name, task_cfg in tasks.items():
        if not str(task_name).strip() or not isinstance(task_cfg, dict):
            raise ValueError("Invalid falsification task definition")
        values = task_cfg.get("values")
        if not isinstance(values, dict) or set(values) != set(SHIFT_AXES):
            raise ValueError(f"Task {task_name} must define every falsification axis")
        for axis in SHIFT_AXES:
            assignment = values[axis]
            if not isinstance(assignment, dict):
                raise ValueError(f"Task {task_name} axis {axis} must be an object")
            source = _unique_strings(
                assignment.get("source"),
                field=f"{task_name}.{axis}.source",
                allow_empty=False,
            )
            heldout = _unique_strings(
                assignment.get("heldout", []),
                field=f"{task_name}.{axis}.heldout",
                allow_empty=True,
            )
            overlap = set(source).intersection(heldout)
            if overlap:
                raise ValueError(
                    f"Task {task_name} axis {axis} assigns {sorted(overlap)} twice"
                )
        hard_negative = task_cfg.get("hard_negative")
        if not isinstance(hard_negative, dict) or not isinstance(
            hard_negative.get("enabled"), bool
        ):
            raise ValueError(f"Task {task_name} requires a hard_negative definition")
        triggers = _unique_strings(
            hard_negative.get("trigger_conditions", []),
            field=f"{task_name}.hard_negative.trigger_conditions",
            allow_empty=not hard_negative["enabled"],
        )
        if hard_negative["enabled"] and not triggers:
            raise ValueError(
                f"Task {task_name} enables hard negatives without triggers"
            )
        if (
            not hard_negative["enabled"]
            and not str(hard_negative.get("blocked_reason", "")).strip()
        ):
            raise ValueError(
                f"Disabled hard negatives for {task_name} need blocked_reason"
            )

    behavior = registry.get("behavior_transfer")
    if not isinstance(behavior, dict):
        raise ValueError("Falsification registry requires behavior_transfer")
    source_behaviors = _unique_strings(
        behavior.get("source_values"),
        field="behavior_transfer.source_values",
        allow_empty=False,
    )
    heldout_behaviors = _unique_strings(
        behavior.get("heldout_values"),
        field="behavior_transfer.heldout_values",
        allow_empty=False,
    )
    known_behaviors = {
        value
        for task_cfg in tasks.values()
        for value in task_cfg["values"]["behavior"]["source"]
    }
    if not set(source_behaviors + heldout_behaviors).issubset(known_behaviors):
        raise ValueError("Behavior transfer references an unregistered task behavior")
    required_axes = set(
        _unique_strings(
            registry.get("required_final_heldout_axes"),
            field="required_final_heldout_axes",
            allow_empty=False,
        )
    )
    if required_axes != set(SHIFT_AXES):
        raise ValueError(
            "Final falsification coverage must require all registered axes"
        )


def _axis_role(task_cfg: dict[str, Any], axis: str, value: str) -> str:
    assignment = task_cfg["values"][axis]
    if value in assignment["source"]:
        return "source"
    if value in assignment["heldout"]:
        return "heldout"
    raise ValueError(f"Unregistered {axis} value {value!r}")


def _metadata_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "metadata_sha256"}


def make_falsification_metadata(
    *,
    registry: dict[str, Any],
    registry_sha256: str,
    task_name: str,
    axis_values: dict[str, str],
    transformation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not SHA256_RE.fullmatch(registry_sha256):
        raise ValueError("Invalid falsification registry SHA-256")
    if task_name not in registry["tasks"]:
        raise ValueError(
            f"Task {task_name!r} is absent from the falsification registry"
        )
    if set(axis_values) != set(SHIFT_AXES):
        raise ValueError(f"Falsification metadata requires axes {SHIFT_AXES}")
    task_cfg = registry["tasks"][task_name]
    axes = {
        axis: {
            "value": str(axis_values[axis]),
            "role": _axis_role(task_cfg, axis, str(axis_values[axis])),
        }
        for axis in SHIFT_AXES
    }
    metadata: dict[str, Any] = {
        "schema_version": FALSIFICATION_EXAMPLE_SCHEMA_VERSION,
        "registry_id": registry["registry_id"],
        "registry_sha256": registry_sha256,
        "axes": axes,
    }
    if transformation is not None:
        metadata["transformation"] = transformation
    metadata["metadata_sha256"] = content_hash(_metadata_payload(metadata))
    validate_falsification_metadata(
        metadata,
        registry=registry,
        registry_sha256=registry_sha256,
        task_name=task_name,
    )
    return metadata


def _validate_transformation(
    transformation: Any,
    *,
    metadata: dict[str, Any],
    registry: dict[str, Any],
) -> None:
    if not isinstance(transformation, dict):
        raise ValueError(
            "Held-out presentation/obfuscation requires transformation provenance"
        )
    axis = str(transformation.get("axis", ""))
    if axis not in TRANSFORMED_AXES:
        raise ValueError("Transformation axis must be presentation or obfuscation")
    if metadata["axes"][axis]["role"] != "heldout":
        raise ValueError("Transformation axis must be assigned heldout")
    for key in (
        "parent_scenario_id",
        "variant_id",
        "generator_protocol",
        "generator_id",
        "transformation_source",
        "parent_prompt_sha256",
        "variant_prompt_sha256",
        "source_payload_sha256",
        "answer_metadata_sha256",
    ):
        if not str(transformation.get(key, "")).strip():
            raise ValueError(f"Transformation provenance requires {key}")
    for key in (
        "parent_prompt_sha256",
        "variant_prompt_sha256",
        "source_payload_sha256",
        "answer_metadata_sha256",
    ):
        if not SHA256_RE.fullmatch(str(transformation[key])):
            raise ValueError(f"Transformation {key} must be a SHA-256")
    validation = registry["transformation_validation"]
    registered = validation["allowed_generators"][axis]
    axis_value = str(metadata["axes"][axis]["value"])
    if transformation["generator_id"] != axis_value:
        raise ValueError("Transformation generator_id must match the held-out value")
    if registered.get(axis_value) != transformation["generator_protocol"]:
        raise ValueError("Transformation uses an unregistered deterministic generator")
    if transformation.get("transformation_source") != "executable_generator":
        raise ValueError("Transformation source must be executable_generator")
    for key in (
        "inverse_verified",
        "verbatim_payload_bound",
        "answer_metadata_unchanged",
    ):
        if transformation.get(key) is not True:
            raise ValueError(f"Transformation failed invariant {key}")


def validate_falsification_metadata(
    metadata: Any,
    *,
    registry: dict[str, Any],
    registry_sha256: str,
    task_name: str,
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("Scenario metadata lacks a falsification object")
    if metadata.get("schema_version") != FALSIFICATION_EXAMPLE_SCHEMA_VERSION:
        raise ValueError("Unsupported falsification-example schema")
    if (
        metadata.get("registry_id") != registry["registry_id"]
        or metadata.get("registry_sha256") != registry_sha256
    ):
        raise ValueError("Falsification metadata does not match the frozen registry")
    if metadata.get("metadata_sha256") != content_hash(_metadata_payload(metadata)):
        raise ValueError("Falsification metadata hash mismatch")
    axes = metadata.get("axes")
    if not isinstance(axes, dict) or set(axes) != set(SHIFT_AXES):
        raise ValueError("Falsification metadata has incomplete axes")
    if task_name not in registry["tasks"]:
        raise ValueError(
            f"Task {task_name!r} is absent from the falsification registry"
        )
    for axis in SHIFT_AXES:
        entry = axes[axis]
        if not isinstance(entry, dict) or set(entry) != {"value", "role"}:
            raise ValueError(f"Invalid falsification axis entry {axis}")
        value = str(entry["value"])
        expected_role = _axis_role(registry["tasks"][task_name], axis, value)
        if entry["role"] != expected_role:
            raise ValueError(
                f"Falsification axis {axis} has an incorrect source/heldout role"
            )
    heldout_transformed = [
        axis for axis in TRANSFORMED_AXES if axes[axis]["role"] == "heldout"
    ]
    if heldout_transformed:
        if len(heldout_transformed) != 1:
            raise ValueError(
                "Each deterministic variant must isolate one transformed shift axis"
            )
        _validate_transformation(
            metadata.get("transformation"), metadata=metadata, registry=registry
        )
        if metadata["transformation"]["axis"] not in heldout_transformed:
            raise ValueError(
                "Transformation provenance does not match the held-out axis"
            )
    elif "transformation" in metadata:
        raise ValueError(
            "Source examples cannot carry held-out transformation provenance"
        )
    return metadata


def falsification_signature(metadata: dict[str, Any]) -> str:
    axes = metadata.get("axes") or {}
    return content_hash(
        {axis: (axes.get(axis) or {}).get("value") for axis in SHIFT_AXES}
    )


def has_heldout_shift(metadata: dict[str, Any]) -> bool:
    return any(
        (metadata.get("axes", {}).get(axis) or {}).get("role") == "heldout"
        for axis in SHIFT_AXES
    )


def prompt_messages_sha256(row: dict[str, Any]) -> str:
    messages = row.get("prompt_messages")
    if not isinstance(messages, list):
        all_messages = row.get("messages")
        if isinstance(all_messages, list):
            messages = [
                message
                for message in all_messages
                if isinstance(message, dict) and message.get("role") != "assistant"
            ]
    if not isinstance(messages, list) or not messages:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Cannot hash an empty scenario prompt")
        messages = [{"role": "user", "content": prompt}]
    return content_hash(messages)


def load_falsification_evaluation_manifest(
    path: Path,
    *,
    registry: dict[str, Any],
    registry_sha256: str,
    check_source: bool,
    minimum_hard_negative_groups: int = 0,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing falsification evaluation manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Falsification evaluation manifest {path} must be an object")
    if manifest.get("schema_version") != FALSIFICATION_EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported falsification evaluation schema in {path}")
    if (
        manifest.get("registry_id") != registry["registry_id"]
        or manifest.get("registry_sha256") != registry_sha256
    ):
        raise ValueError(
            f"Falsification evaluation manifest {path} uses the wrong registry"
        )
    expected_hash = content_hash(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != expected_hash:
        raise ValueError(f"Falsification evaluation manifest {path} failed its hash")
    if not SHA256_RE.fullmatch(str(manifest.get("manifest_id", ""))):
        raise ValueError(
            f"Falsification evaluation manifest {path} has invalid manifest_id"
        )
    task_name = str(manifest.get("task_name", ""))
    if task_name not in registry["tasks"]:
        raise ValueError(f"Falsification evaluation manifest {path} has unknown task")
    if not str(manifest.get("model", "")).strip():
        raise ValueError(f"Falsification evaluation manifest {path} lacks model name")
    for field in (
        "monitored_model_id",
        "monitored_model_revision",
        "monitored_tokenizer_revision",
    ):
        if not str(manifest.get(field, "")).strip():
            raise ValueError(f"Falsification evaluation manifest {path} lacks {field}")
    if not re.fullmatch(
        r"[0-9a-f]{7,64}", str(manifest.get("monitored_model_revision", ""))
    ) or not re.fullmatch(
        r"[0-9a-f]{7,64}", str(manifest.get("monitored_tokenizer_revision", ""))
    ):
        raise ValueError(
            f"Falsification evaluation manifest {path} has moving revisions"
        )
    if manifest.get("hard_negative_match_type") != "exact_trigger_prompt":
        raise ValueError(
            f"Falsification evaluation manifest {path} has wrong match type"
        )
    if not SHA256_RE.fullmatch(str(manifest.get("source_data_sha256", ""))):
        raise ValueError(
            f"Falsification evaluation manifest {path} has invalid source hash"
        )
    expected_manifest_id = content_hash(
        {
            "registry_sha256": registry_sha256,
            "task_name": task_name,
            "source_data_sha256": manifest["source_data_sha256"],
            "monitored_model_revision": manifest["monitored_model_revision"],
        }
    )
    if manifest["manifest_id"] != expected_manifest_id:
        raise ValueError(
            f"Falsification evaluation manifest {path} has stale manifest_id"
        )
    if check_source:
        source_path = Path(str(manifest.get("source_data_file", "")))
        if not source_path.exists() or file_sha256(source_path) != manifest.get(
            "source_data_sha256"
        ):
            raise ValueError(
                f"Falsification evaluation manifest {path} has stale source data"
            )
    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"Falsification evaluation manifest {path} has no examples")
    example_map: dict[str, dict[str, Any]] = {}
    group_splits: dict[str, set[str]] = {}
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError(
                f"Falsification evaluation manifest {path} has invalid examples"
            )
        example_id = str(example.get("example_id", ""))
        if not example_id or example_id in example_map:
            raise ValueError(
                f"Falsification evaluation manifest {path} duplicates example IDs"
            )
        if example.get("label") not in {0, 1}:
            raise ValueError(
                f"Falsification evaluation manifest {path} has non-binary labels"
            )
        for hash_key in ("prompt_sha256", "row_sha256", "shift_signature"):
            if not SHA256_RE.fullmatch(str(example.get(hash_key, ""))):
                raise ValueError(
                    f"Falsification evaluation manifest {path} has invalid {hash_key}"
                )
        axes = example.get("axes")
        if not isinstance(axes, dict) or set(axes) != set(SHIFT_AXES):
            raise ValueError(
                f"Falsification evaluation manifest {path} has incomplete axes"
            )
        heldout = False
        for axis in SHIFT_AXES:
            entry = axes[axis]
            if not isinstance(entry, dict) or set(entry) != {"value", "role"}:
                raise ValueError(
                    f"Falsification evaluation manifest {path} has invalid axis {axis}"
                )
            expected_role = _axis_role(
                registry["tasks"][task_name], axis, str(entry["value"])
            )
            if entry["role"] != expected_role:
                raise ValueError(
                    f"Falsification evaluation manifest {path} has incorrect axis role"
                )
            heldout = heldout or expected_role == "heldout"
        split = str(example.get("protocol_split", ""))
        if split not in {"train", "calibration", "eval", "test"}:
            raise ValueError(
                f"Falsification evaluation manifest {path} has invalid split"
            )
        if heldout and split != "test":
            raise ValueError(
                f"Falsification evaluation manifest {path} leaks held-out shifts"
            )
        group_id = str(example.get("group_id", ""))
        if not group_id:
            raise ValueError(
                f"Falsification evaluation manifest {path} lacks group IDs"
            )
        group_splits.setdefault(group_id, set()).add(split)
        example_map[example_id] = example
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError(
            f"Falsification evaluation manifest {path} leaks groups across splits"
        )

    pairs = manifest.get("hard_negative_pairs")
    if not isinstance(pairs, list):
        raise ValueError(
            f"Falsification evaluation manifest {path} lacks hard-negative pairs"
        )
    pair_ids: set[str] = set()
    for pair in pairs:
        if (
            not isinstance(pair, dict)
            or pair.get("match_type") != "exact_trigger_prompt"
        ):
            raise ValueError(
                f"Falsification evaluation manifest {path} has invalid pairs"
            )
        pair_id = str(pair.get("pair_id", ""))
        if not SHA256_RE.fullmatch(pair_id) or pair_id in pair_ids:
            raise ValueError(
                f"Falsification evaluation manifest {path} duplicates pair IDs"
            )
        positive = example_map.get(str(pair.get("positive_example_id", "")))
        negative = example_map.get(str(pair.get("negative_example_id", "")))
        if positive is None or negative is None:
            raise ValueError(
                f"Falsification evaluation manifest {path} references absent examples"
            )
        if positive["label"] != 1 or negative["label"] != 0:
            raise ValueError(
                f"Falsification evaluation manifest {path} reverses pair labels"
            )
        for key in ("scenario_id", "group_id", "shift_signature", "prompt_sha256"):
            if positive[key] != negative[key] or pair.get(key) != positive[key]:
                raise ValueError(
                    f"Falsification evaluation manifest {path} has unmatched pair {pair_id}"
                )
        if (
            pair.get("positive_row_sha256") != positive["row_sha256"]
            or pair.get("negative_row_sha256") != negative["row_sha256"]
        ):
            raise ValueError(
                f"Falsification evaluation manifest {path} has stale pair rows"
            )
        pair_payload = {
            "task_name": task_name,
            "scenario_id": pair["scenario_id"],
            "group_id": pair["group_id"],
            "shift_signature": pair["shift_signature"],
            "prompt_sha256": pair["prompt_sha256"],
            "positive_example_id": pair["positive_example_id"],
            "negative_example_id": pair["negative_example_id"],
            "positive_row_sha256": pair["positive_row_sha256"],
            "negative_row_sha256": pair["negative_row_sha256"],
        }
        if pair_id != content_hash(pair_payload):
            raise ValueError(
                f"Falsification evaluation manifest {path} has stale pair hash"
            )
        pair_ids.add(pair_id)
    n_groups = len({str(pair["group_id"]) for pair in pairs})
    if n_groups < minimum_hard_negative_groups:
        raise ValueError(
            f"Falsification evaluation manifest {path} has {n_groups} hard-negative "
            f"groups; requires {minimum_hard_negative_groups}"
        )
    summary = manifest.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("n_examples") != len(examples)
        or summary.get("n_hard_negative_pairs") != len(pairs)
        or summary.get("n_hard_negative_groups") != n_groups
    ):
        raise ValueError(
            f"Falsification evaluation manifest {path} has a stale summary"
        )
    return manifest
