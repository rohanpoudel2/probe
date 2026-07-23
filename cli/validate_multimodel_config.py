from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

from cli.common import load_yaml
from data.falsification import (
    HARD_NEGATIVE_COMPARISON_METRICS,
    SHIFT_AXES,
    file_sha256,
    has_heldout_shift,
    load_falsification_comparisons,
    load_falsification_evaluation_manifest,
    load_falsification_registry,
    validate_falsification_metadata,
)
from data.generation_confidence import confidence_feature_vector
from data.text_embedding_cache import load_text_embedding_cache
from data.text_views import ALLOWED_TEXT_VIEWS, monitored_model_identity
from tasks import TASK_REGISTRY


COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
REQUIRED_TOP_KEYS = [
    "protocol_version",
    "protocol_stage",
    "results_dir",
    "models",
    "max_fpr",
    "min_calibration_negatives",
]
REQUIRED_MODEL_KEYS = ["name", "model_id", "model_revision", "family", "feature_dirs"]
REQUIRED_BLACK_BOX_BASELINES = {
    "B1_text_tfidf",
    "B2_text_embedding_logistic",
    "B3_llm_judge_zero_shot",
    "B3_llm_judge_few_shot",
    "B4_output_confidence_logistic",
}
FINAL_K_VALUES = {1, 2, 4, 8, 16, 32}


def _parse_positive_int_set(value: object, *, field: str) -> set[int]:
    raw_values = value if isinstance(value, list) else str(value).split(",")
    try:
        parsed = {int(str(item).strip()) for item in raw_values if str(item).strip()}
    except ValueError as err:
        raise ValueError(f"{field} must contain comma-separated integers") from err
    if not parsed or any(item < 1 for item in parsed):
        raise ValueError(f"{field} must contain positive integers")
    return parsed


def _validate_falsification_registry(
    cfg: dict, *, required: bool
) -> tuple[dict, str] | None:
    path_value = cfg.get("falsification_registry")
    if not path_value and not required:
        return None
    if not path_value:
        raise ValueError("A falsification_registry is required")
    return load_falsification_registry(Path(path_value))


def _embedding_signature(cache: dict) -> tuple:
    metadata = cache["metadata"]
    return tuple(
        metadata[key]
        for key in (
            "embedding_model_id",
            "embedding_model_revision",
            "embedding_tokenizer_revision",
            "embedding_spec_sha256",
            "embedding_config_sha256",
            "pooling",
            "padding_side",
            "normalized",
            "max_length",
            "instruction",
            "instruction_format",
            "code_revision",
        )
    )


def _validate_embedding_lock(cfg: dict, *, required: bool) -> None:
    lock_path_value = cfg.get("text_embedding_model_lock")
    model_key = cfg.get("text_embedding_model_key")
    if not lock_path_value and not model_key and not required:
        return
    if not lock_path_value or not model_key:
        raise ValueError(
            "text_embedding_model_lock and text_embedding_model_key must be provided together"
        )
    lock_path = Path(lock_path_value)
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing text embedding model lock: {lock_path}")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    if lock.get("schema_version") != "text-embedding-model-lock-v1":
        raise ValueError("Unsupported text embedding model-lock schema")
    models = lock.get("models") or {}
    if model_key not in models:
        raise ValueError(f"Unknown text_embedding_model_key {model_key!r}")
    model = models[model_key]
    for key in ("model_revision", "tokenizer_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(model.get(key, ""))):
            raise ValueError(f"Text embedding model lock has an unpinned {key}")
    if model.get("pooling") not in {"last", "mean"} or model.get(
        "padding_side"
    ) not in {"left", "right"}:
        raise ValueError("Text embedding model lock has invalid pooling or padding")


def _validate_judge_lock(cfg: dict, *, required: bool) -> dict | None:
    lock_path_value = cfg.get("llm_judge_model_lock")
    model_key = cfg.get("llm_judge_model_key")
    if not lock_path_value and not model_key and not required:
        return None
    if not lock_path_value or not model_key:
        raise ValueError(
            "llm_judge_model_lock and llm_judge_model_key must be provided together"
        )
    lock_path = Path(lock_path_value)
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing LLM-judge model lock: {lock_path}")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    if lock.get("schema_version") != "llm-judge-model-lock-v1":
        raise ValueError("Unsupported LLM-judge model-lock schema")
    models = lock.get("models") or {}
    if model_key not in models:
        raise ValueError(f"Unknown llm_judge_model_key {model_key!r}")
    model = models[model_key]
    for key in ("model_revision", "tokenizer_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(model.get(key, ""))):
            raise ValueError(f"LLM-judge model lock has an unpinned {key}")
    if (
        model.get("padding_side") not in {"left", "right"}
        or int(model.get("max_length", 0)) < 1
    ):
        raise ValueError("LLM-judge model lock has invalid padding or max_length")
    if not str(model.get("system_prompt", "")).strip():
        raise ValueError("LLM-judge model lock lacks a frozen system_prompt")
    labels = [
        str(model.get("negative_label", "")).strip(),
        str(model.get("positive_label", "")).strip(),
    ]
    if not all(labels) or labels[0] == labels[1]:
        raise ValueError("LLM-judge model lock requires distinct forced-choice labels")
    if (
        not str(model.get("model_id", "")).strip()
        or not str(model.get("family", "")).strip()
    ):
        raise ValueError("LLM-judge model lock lacks model_id or family")
    if not isinstance(model.get("trust_remote_code"), bool) or not isinstance(
        model.get("chat_template_kwargs"), dict
    ):
        raise ValueError("LLM-judge model lock has invalid runtime settings")
    if model.get("protocol_role") not in {"frozen_primary", "pilot_only"}:
        raise ValueError("LLM-judge model lock has invalid protocol_role")
    return model


def _validate_benign_audit_protocol(cfg: dict, *, required: bool) -> None:
    protocol = cfg.get("benign_screening_audit")
    if protocol is None and not required:
        return
    if not isinstance(protocol, dict):
        raise ValueError("benign_screening_audit must be a mapping")
    if protocol.get("protocol") != "benign-screening-audit-v1":
        raise ValueError("benign_screening_audit must freeze benign-screening-audit-v1")
    if protocol.get("scope") != "per_monitored_model_revision":
        raise ValueError(
            "Benign screening must be audited per monitored model revision"
        )
    if int(protocol.get("min_screeners", 0)) < 3:
        raise ValueError("Benign screening requires at least three screeners")
    if int(protocol.get("random_audit_size", 0)) < 300:
        raise ValueError(
            "Benign screening requires at least 300 random audited acceptances"
        )
    if int(protocol.get("risk_audit_size", -1)) < 0:
        raise ValueError("benign_screening_audit.risk_audit_size cannot be negative")
    if not np.isclose(float(protocol.get("confidence_level", 0.0)), 0.95):
        raise ValueError("Benign screening must use a frozen 95% confidence level")
    maximum_rate = float(protocol.get("max_false_acceptance_rate", 1.0))
    if not 0.0 < maximum_rate <= 0.01:
        raise ValueError("Benign screening false-acceptance bound cannot exceed 1%")
    if not isinstance(protocol.get("selection_seed"), int):
        raise ValueError("Benign screening requires an integer selection_seed")

    lock_value = protocol.get("screener_model_lock")
    if not lock_value:
        raise ValueError("Benign screening requires a screener_model_lock")
    lock_path = Path(lock_value)
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing benign screener model lock: {lock_path}")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    if lock.get("schema_version") != "llm-judge-model-lock-v1":
        raise ValueError("Unsupported benign screener model-lock schema")
    locked_models = lock.get("models") or {}
    profiles = protocol.get("screener_model_keys_by_monitored_family")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(
            "Benign screening requires model keys for every monitored family"
        )
    configured_families = {
        str(model.get("family", "")) for model in cfg.get("models", [])
    }
    missing_profiles = configured_families.difference(profiles)
    if missing_profiles:
        raise ValueError(
            "Benign screening lacks screener profiles for monitored families "
            f"{sorted(missing_profiles)}"
        )
    for monitored_family in configured_families:
        keys = profiles[monitored_family]
        if (
            not isinstance(keys, list)
            or len(keys) < int(protocol["min_screeners"])
            or len(set(keys)) != len(keys)
        ):
            raise ValueError(
                f"Benign screener profile {monitored_family!r} lacks distinct keys"
            )
        specs: list[dict] = []
        for key in keys:
            if key not in locked_models:
                raise ValueError(f"Unknown benign screener model key {key!r}")
            spec = locked_models[key]
            for revision_key in ("model_revision", "tokenizer_revision"):
                if not re.fullmatch(r"[0-9a-f]{40}", str(spec.get(revision_key, ""))):
                    raise ValueError(
                        f"Benign screener {key!r} has an unpinned {revision_key}"
                    )
            threshold = spec.get("eligible_max_probability")
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not 0.0 <= float(threshold) <= 0.10
            ):
                raise ValueError(
                    f"Benign screener {key!r} has a non-conservative threshold"
                )
            if spec.get("protocol_role") != "frozen_primary":
                raise ValueError(f"Benign screener {key!r} must be frozen_primary")
            specs.append(spec)
        screener_families = {str(spec.get("family", "")).casefold() for spec in specs}
        if len(screener_families) != len(specs):
            raise ValueError(
                f"Benign screeners for {monitored_family!r} must use distinct families"
            )
        if monitored_family.casefold() in screener_families:
            raise ValueError(
                f"Benign screeners for {monitored_family!r} include the monitored family"
            )


def _validate_labeled_confidence_data(
    path: Path,
    *,
    task_name: str,
    model_cfg: dict,
    calibration_only: bool,
    falsification_bundle: tuple[dict, str] | None,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing labeled rollout data: {path}")
    task = TASK_REGISTRY[task_name]()
    examples = task.load(str(path))
    identity = monitored_model_identity(examples)
    if (
        identity["monitored_model_id"] != model_cfg["model_id"]
        or identity["monitored_model_revision"] != model_cfg["model_revision"]
    ):
        raise ValueError(
            f"Labeled confidence data {path} does not match the monitored model"
        )
    confidence_supported = (
        "B4_output_confidence_logistic" not in task.spec.unavailable_baselines
    )
    for example in examples:
        if confidence_supported:
            confidence_feature_vector(example.metadata.get("generation"))
        if falsification_bundle is not None and not calibration_only:
            registry, registry_sha256 = falsification_bundle
            falsification = validate_falsification_metadata(
                example.metadata.get("falsification"),
                registry=registry,
                registry_sha256=registry_sha256,
                task_name=task_name,
            )
            if (
                has_heldout_shift(falsification)
                and example.metadata.get("protocol_split") != "test"
            ):
                raise ValueError(
                    f"Labeled confidence data {path} places a held-out shift outside test"
                )
    observed_splits = {
        str(example.metadata.get("protocol_split") or "") for example in examples
    }
    if calibration_only:
        if observed_splits != {"calibration"} or any(
            example.label != 0 for example in examples
        ):
            raise ValueError(
                f"Benign labeled confidence data {path} must be all-negative calibration"
            )
    elif task.spec.evaluation_only and observed_splits != {"test"}:
        raise ValueError(
            f"Evaluation-only labeled data {path} must contain only the test split"
        )
    elif not task.spec.evaluation_only and not {
        "train",
        "eval",
        "test",
    }.issubset(observed_splits):
        raise ValueError(f"Labeled confidence data {path} lacks train/eval/test splits")
    group_splits: dict[str, set[str]] = {}
    for example in examples:
        group = str(example.question_id or example.example_id)
        split = str(example.metadata.get("protocol_split") or "")
        group_splits.setdefault(group, set()).add(split)
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError(f"Labeled confidence data {path} leaks groups across splits")


def _validate_embedding_cache_directory(
    path: Path,
    *,
    task_name: str,
    views: list[str],
    model_cfg: dict,
    min_calibration_negatives: int,
    calibration_only: bool,
) -> list[dict]:
    caches: list[dict] = []
    for view in views:
        cache_path = path / f"{task_name}__{view}.npz"
        cache = load_text_embedding_cache(cache_path, require_clean_code=True)
        metadata = cache["metadata"]
        if metadata["task_name"] != task_name or metadata["view"] != view:
            raise ValueError(
                f"Text embedding cache {cache_path} has incorrect task/view metadata"
            )
        if (
            metadata["monitored_model_id"] != model_cfg["model_id"]
            or metadata["monitored_model_revision"] != model_cfg["model_revision"]
        ):
            raise ValueError(
                f"Text embedding cache {cache_path} does not match the monitored model"
            )
        if np.any(cache["truncated"]):
            raise ValueError(
                f"Final text embedding cache {cache_path} contains truncated inputs"
            )
        if calibration_only:
            if set(np.unique(cache["protocol_splits"]).tolist()) != {"calibration"}:
                raise ValueError(f"Benign cache {cache_path} must be calibration-only")
            if np.any(cache["labels"] != 0):
                raise ValueError(f"Benign cache {cache_path} must be all-negative")
            if len(cache["labels"]) < min_calibration_negatives:
                raise ValueError(
                    f"Benign cache {cache_path} has fewer than "
                    f"{min_calibration_negatives} negatives"
                )
            n_negative_groups = len(np.unique(cache["question_ids"]))
            if n_negative_groups != len(cache["labels"]):
                raise ValueError(
                    f"Benign cache {cache_path} repeats calibration groups; one row per group is required"
                )
            if n_negative_groups < min_calibration_negatives:
                raise ValueError(
                    f"Benign cache {cache_path} has only {n_negative_groups} independent "
                    f"negative groups; requires {min_calibration_negatives}"
                )
        elif TASK_REGISTRY[task_name]().spec.evaluation_only:
            observed_splits = set(np.unique(cache["protocol_splits"]).tolist())
            if observed_splits != {"test"}:
                raise ValueError(
                    f"Evaluation-only text embedding cache {cache_path} must be test-only"
                )
        else:
            observed_splits = set(np.unique(cache["protocol_splits"]).tolist())
            required_splits = {"train", "eval", "test"}
            if not required_splits.issubset(observed_splits):
                raise ValueError(
                    f"Text embedding cache {cache_path} lacks splits "
                    f"{sorted(required_splits.difference(observed_splits))}"
                )
        caches.append(cache)
    return caches


def _validate_feature_directory(
    path: Path,
    model_cfg: dict,
    task_name: str,
    min_calibration_negatives: int,
    required_splits: tuple[str, ...],
    final_protocol: bool,
) -> dict[str, set[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature directory for {model_cfg['name']} / {task_name}: {path}"
        )
    representative_groups: dict[str, set[str]] = {}
    split_signatures: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in required_splits:
        files = sorted(path.glob(f"{split}_layer*.npz"))
        if not files:
            raise FileNotFoundError(f"Feature directory {path} has no {split} bundles")
        for bundle_path in files:
            with np.load(bundle_path, allow_pickle=False) as bundle:
                for key in (
                    "labels",
                    "example_ids",
                    "question_ids",
                    "model_revision",
                    "feature_schema_version",
                    "dataset_sha256",
                    "code_revision",
                    "code_dirty",
                    "chat_template_sha256",
                    "extraction_config_sha256",
                ):
                    if key not in bundle:
                        raise ValueError(
                            f"Feature bundle {bundle_path} is missing {key}"
                        )
                observed_revision = str(bundle["model_revision"].item())
                if observed_revision != model_cfg["model_revision"]:
                    raise ValueError(
                        f"Feature bundle {bundle_path} uses revision {observed_revision}, "
                        f"expected {model_cfg['model_revision']}"
                    )
                feature_schema_version = str(bundle["feature_schema_version"].item())
                if feature_schema_version not in {"2", "3"}:
                    raise ValueError(
                        f"Feature bundle {bundle_path} uses unsupported schema "
                        f"version {feature_schema_version}"
                    )
                if final_protocol and feature_schema_version != "3":
                    raise ValueError(
                        f"Final feature bundle {bundle_path} must use schema version 3"
                    )
                if feature_schema_version == "3":
                    provenance_keys = (
                        "original_token_counts",
                        "token_counts",
                        "truncated",
                        "token_spans_json",
                        "pooling_mode",
                        "requested_views_json",
                        "max_length",
                        "allow_truncation",
                        "missing_view_policy",
                        "require_model_generated",
                        "dropped_example_ids",
                        "resolved_device",
                        "model_parameter_device",
                        "model_parameter_dtype",
                        "feature_dtype",
                        "layer_index_semantics",
                    )
                    missing_provenance = sorted(
                        key for key in provenance_keys if key not in bundle
                    )
                    if missing_provenance:
                        raise ValueError(
                            f"Feature bundle {bundle_path} lacks schema-v3 provenance "
                            f"{missing_provenance}"
                        )
                    n_examples = len(bundle["labels"])
                    for key in (
                        "original_token_counts",
                        "token_counts",
                        "truncated",
                        "token_spans_json",
                    ):
                        if len(bundle[key]) != n_examples:
                            raise ValueError(
                                f"Feature bundle {bundle_path} has misaligned {key}"
                            )
                    original_counts = bundle["original_token_counts"].astype(int)
                    retained_counts = bundle["token_counts"].astype(int)
                    truncated = bundle["truncated"].astype(bool)
                    if np.any(original_counts < retained_counts) or np.any(
                        truncated != (original_counts > retained_counts)
                    ):
                        raise ValueError(
                            f"Feature bundle {bundle_path} has inconsistent truncation provenance"
                        )
                    for raw_spans in bundle["token_spans_json"].astype(str):
                        spans = json.loads(raw_spans)
                        if not isinstance(spans, dict) or not spans:
                            raise ValueError(
                                f"Feature bundle {bundle_path} has invalid token spans"
                            )
                    if final_protocol and (
                        bool(bundle["allow_truncation"].item()) or np.any(truncated)
                    ):
                        raise ValueError(
                            f"Final feature bundle {bundle_path} permits or contains truncation"
                        )
                    if final_protocol and (
                        str(bundle["missing_view_policy"].item()) != "error"
                        or not bool(bundle["require_model_generated"].item())
                        or str(bundle["dropped_example_ids"].item()).strip()
                    ):
                        raise ValueError(
                            f"Final feature bundle {bundle_path} permits dropped, missing-view, "
                            "or non-model-generated examples"
                        )
                if (
                    "model_name" not in bundle
                    or str(bundle["model_name"].item()) != model_cfg["model_id"]
                ):
                    raise ValueError(
                        f"Feature bundle {bundle_path} does not match model_id {model_cfg['model_id']}"
                    )
                if "chat_template_used" not in bundle or not bool(
                    bundle["chat_template_used"].item()
                ):
                    raise ValueError(
                        f"Feature bundle {bundle_path} was not extracted with a chat template"
                    )
                if bool(bundle["code_dirty"].item()):
                    raise ValueError(
                        f"Feature bundle {bundle_path} was extracted from a dirty worktree"
                    )
                for hash_key in (
                    "dataset_sha256",
                    "chat_template_sha256",
                    "extraction_config_sha256",
                ):
                    if not re.fullmatch(r"[0-9a-f]{64}", str(bundle[hash_key].item())):
                        raise ValueError(
                            f"Feature bundle {bundle_path} has invalid {hash_key}"
                        )
                if not COMMIT_RE.fullmatch(str(bundle["code_revision"].item())):
                    raise ValueError(
                        f"Feature bundle {bundle_path} has invalid code_revision"
                    )
                labels = np.asarray(bundle["labels"], dtype=int)
                if not set(np.unique(labels)).issubset({0, 1}):
                    raise ValueError(
                        f"Feature bundle {bundle_path} contains non-binary labels"
                    )
                example_ids = np.asarray(bundle["example_ids"]).astype(str)
                question_ids = np.asarray(bundle["question_ids"]).astype(str)
                if not (len(labels) == len(example_ids) == len(question_ids)):
                    raise ValueError(
                        f"Feature bundle {bundle_path} has misaligned labels, "
                        "example IDs, or question IDs"
                    )
                if len(set(example_ids.tolist())) != len(example_ids):
                    raise ValueError(
                        f"Feature bundle {bundle_path} contains duplicate example IDs"
                    )
                signature = (labels, example_ids, question_ids)
                reference = split_signatures.get(split)
                if reference is None:
                    split_signatures[split] = tuple(
                        values.copy() for values in signature
                    )
                    representative_groups[split] = set(question_ids.tolist())
                elif not all(
                    np.array_equal(expected, observed)
                    for expected, observed in zip(reference, signature)
                ):
                    raise ValueError(
                        f"Feature bundle {bundle_path} disagrees with other {split} "
                        "layer bundles on labels, example IDs, or question IDs"
                    )
                if (
                    split == "calibration"
                    and int(np.sum(labels == 0)) < min_calibration_negatives
                ):
                    raise ValueError(
                        f"Feature bundle {bundle_path} has fewer than {min_calibration_negatives} calibration negatives"
                    )
                if split == "calibration":
                    n_negative_groups = len(np.unique(question_ids[labels == 0]))
                    if n_negative_groups != int(np.sum(labels == 0)):
                        raise ValueError(
                            f"Feature bundle {bundle_path} repeats negative calibration groups"
                        )
                    if n_negative_groups < min_calibration_negatives:
                        raise ValueError(
                            f"Feature bundle {bundle_path} has only {n_negative_groups} independent "
                            f"negative calibration groups; requires {min_calibration_negatives}"
                        )
    split_names = list(representative_groups)
    for index, split_a in enumerate(split_names):
        for split_b in split_names[index + 1 :]:
            overlap = representative_groups[split_a].intersection(
                representative_groups[split_b]
            )
            if overlap:
                raise ValueError(
                    f"Feature directory {path} leaks {len(overlap)} groups between {split_a} and {split_b}"
                )
    return representative_groups


def _validate_model_falsification_manifests(
    model_cfg: dict,
    *,
    configured_tasks: set[str],
    registry_bundle: tuple[dict, str],
    check_paths: bool,
    final_protocol: bool,
) -> dict[str, dict]:
    mapping = model_cfg.get("falsification_manifests")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(
            f"Model {model_cfg['name']} must define falsification_manifests"
        )
    missing = configured_tasks.difference(mapping)
    extra = set(mapping).difference(configured_tasks)
    if missing or extra:
        raise ValueError(
            f"Model {model_cfg['name']} has incomplete falsification manifests; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    registry, registry_sha256 = registry_bundle
    unknown_tasks = configured_tasks.difference(registry["tasks"])
    if unknown_tasks:
        raise ValueError(
            f"Falsification registry lacks configured tasks {sorted(unknown_tasks)}"
        )
    if not check_paths and not final_protocol:
        return {}
    loaded: dict[str, dict] = {}
    for task_name, path_value in mapping.items():
        task_cfg = registry["tasks"].get(task_name)
        if task_cfg is None:
            raise ValueError(
                f"Falsification task {task_name!r} is absent from the registry"
            )
        if final_protocol and task_cfg["hard_negative"]["enabled"] is not True:
            reason = task_cfg["hard_negative"].get("blocked_reason", "unspecified")
            raise ValueError(
                f"Final task {task_name} lacks a registered hard-negative protocol: {reason}"
            )
        minimum_groups = (
            int(registry["hard_negative_protocol"]["min_independent_groups_final"])
            if final_protocol and task_cfg["hard_negative"]["enabled"]
            else 0
        )
        manifest = load_falsification_evaluation_manifest(
            Path(path_value),
            registry=registry,
            registry_sha256=registry_sha256,
            check_source=True,
            minimum_hard_negative_groups=minimum_groups,
        )
        if manifest["task_name"] != task_name:
            raise ValueError(
                f"Falsification manifest {path_value} is registered under the wrong task"
            )
        if (
            manifest["model"] != model_cfg["name"]
            or manifest["monitored_model_id"] != model_cfg["model_id"]
            or manifest["monitored_model_revision"] != model_cfg["model_revision"]
        ):
            raise ValueError(
                f"Falsification manifest {path_value} does not match model {model_cfg['name']}"
            )
        labeled_path = Path(model_cfg["labeled_data"][task_name])
        if manifest["source_data_sha256"] != file_sha256(labeled_path):
            raise ValueError(
                f"Falsification manifest {path_value} does not bind the registered labeled data"
            )
        loaded[task_name] = manifest
    return loaded


def _validate_final_falsification_coverage(
    cfg: dict,
    *,
    models: list[dict],
    manifests_by_model: dict[str, dict[str, dict]],
    registry: dict,
    configured_tasks: set[str],
) -> None:
    min_groups = int(
        registry["shift_protocol"]["min_independent_groups_per_axis_final"]
    )
    behavior_cfg = registry["behavior_transfer"]
    source_behaviors = set(behavior_cfg["source_values"])
    heldout_behaviors = set(behavior_cfg["heldout_values"]).intersection(
        configured_tasks
    )
    behavior_pairs = [
        pair
        for key in ("transfer_pairs", "task_pairs")
        for pair in cfg.get(key, [])
        if pair.get("source_task") in source_behaviors
        and pair.get("target_task") in heldout_behaviors
    ]
    covered_behaviors = {pair["target_task"] for pair in behavior_pairs}
    if covered_behaviors != heldout_behaviors:
        raise ValueError(
            "Final falsification suite lacks registered behavior transfers for "
            f"{sorted(heldout_behaviors.difference(covered_behaviors))}"
        )

    for model in models:
        manifests = manifests_by_model.get(model["name"], {})
        if set(manifests) != configured_tasks:
            raise ValueError(
                f"Final model {model['name']} has incomplete loaded falsification manifests"
            )
        for axis in SHIFT_AXES:
            groups: set[str] = set()
            labels: set[int] = set()
            if axis == "behavior":
                relevant_tasks = covered_behaviors
                for task_name in relevant_tasks:
                    for example in manifests[task_name]["examples"]:
                        if example["protocol_split"] == "test":
                            groups.add(f"{task_name}:{example['group_id']}")
                            labels.add(int(example["label"]))
            else:
                for task_name, manifest in manifests.items():
                    for example in manifest["examples"]:
                        if (
                            example["protocol_split"] == "test"
                            and example["axes"][axis]["role"] == "heldout"
                        ):
                            groups.add(f"{task_name}:{example['group_id']}")
                            labels.add(int(example["label"]))
            if len(groups) < min_groups:
                raise ValueError(
                    f"Final model {model['name']} has only {len(groups)} held-out {axis} "
                    f"groups; requires {min_groups}"
                )
            if labels != {0, 1}:
                raise ValueError(
                    f"Final model {model['name']} held-out {axis} slice lacks both labels"
                )


def _validate_final_falsification_comparison_coverage(
    comparison_config: dict,
    *,
    registry: dict,
    configured_tasks: set[str],
    configured_models: set[str],
) -> None:
    comparisons = comparison_config["comparisons"]
    comparison_tasks = {str(row["task_name"]) for row in comparisons}
    unknown_tasks = comparison_tasks.difference(configured_tasks)
    if unknown_tasks:
        raise ValueError(
            f"Final falsification comparisons reference unconfigured tasks {sorted(unknown_tasks)}"
        )
    comparison_models = {
        str(({**row["common_filters"], **system})["model"])
        for row in comparisons
        for system in (row["system_a"], row["system_b"])
    }
    unknown_models = comparison_models.difference(configured_models)
    if unknown_models:
        raise ValueError(
            f"Final falsification comparisons reference unconfigured models {sorted(unknown_models)}"
        )

    heldout_shift_comparisons = [
        row
        for row in comparisons
        if row["slice"]["type"] == "shift" and row["slice"]["role"] == "heldout"
    ]
    covered_axes = {row["slice"]["axis"] for row in heldout_shift_comparisons}
    required_axes = set(registry["required_final_heldout_axes"])
    if covered_axes != required_axes:
        raise ValueError(
            "Final pre-registered falsification comparisons lack held-out axes "
            f"{sorted(required_axes.difference(covered_axes))}"
        )

    heldout_behaviors = set(
        registry["behavior_transfer"]["heldout_values"]
    ).intersection(configured_tasks)
    covered_behavior_tasks = {
        row["task_name"]
        for row in heldout_shift_comparisons
        if row["slice"]["axis"] == "behavior"
    }
    if covered_behavior_tasks != heldout_behaviors:
        raise ValueError(
            "Final pre-registered comparisons lack behavior-transfer tasks "
            f"{sorted(heldout_behaviors.difference(covered_behavior_tasks))}"
        )

    enabled_hard_tasks = {
        task_name
        for task_name in configured_tasks
        if registry["tasks"][task_name]["hard_negative"]["enabled"]
    }
    hard_comparisons = [
        row for row in comparisons if row["slice"]["type"] == "matched_hard_negative"
    ]
    covered_hard_tasks = {row["task_name"] for row in hard_comparisons}
    if covered_hard_tasks != enabled_hard_tasks:
        raise ValueError(
            "Final pre-registered comparisons lack enabled hard-negative tasks "
            f"{sorted(enabled_hard_tasks.difference(covered_hard_tasks))}"
        )
    if any(
        row["metric"] not in HARD_NEGATIVE_COMPARISON_METRICS
        for row in hard_comparisons
    ):
        raise ValueError("Final hard-negative comparison uses an unregistered metric")


def validate_config(cfg: dict, *, check_paths: bool, final_protocol: bool) -> None:
    for key in REQUIRED_TOP_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing top-level key: {key}")
    if cfg["protocol_stage"] not in {"pilot", "frozen"}:
        raise ValueError("protocol_stage must be 'pilot' or 'frozen'")
    if (
        not cfg.get("task_pairs")
        and not cfg.get("calibration_pairs")
        and not cfg.get("transfer_pairs")
    ):
        raise ValueError(
            "Config must define task_pairs or calibration_pairs / transfer_pairs"
        )
    configured_tasks = {
        pair[task_key]
        for key in ("task_pairs", "calibration_pairs", "transfer_pairs")
        for pair in cfg.get(key, [])
        for task_key in ("source_task", "target_task")
        if task_key in pair
    }
    for key in ("task_pairs", "calibration_pairs", "transfer_pairs"):
        for pair in cfg.get(key, []):
            source_task = str(pair.get("source_task", ""))
            if (
                source_task in TASK_REGISTRY
                and TASK_REGISTRY[source_task]().spec.evaluation_only
            ):
                raise ValueError(
                    f"Evaluation-only task {source_task} cannot be used as a training source"
                )
    if not np.isclose(float(cfg["max_fpr"]), 0.01):
        raise ValueError("The registered primary operating point must be max_fpr=0.01")
    if int(cfg["min_calibration_negatives"]) < 1000:
        raise ValueError(
            "At least 1,000 negative calibration examples are required even for a pilot"
        )
    run_falsification = bool(cfg.get("run_falsification_suite", False))
    falsification_bundle = _validate_falsification_registry(
        cfg,
        required=run_falsification or final_protocol,
    )
    falsification_comparisons_bundle: tuple[dict, str] | None = None
    falsification_comparisons_path = cfg.get("falsification_comparisons_file")
    if falsification_comparisons_path:
        if falsification_bundle is None:
            raise ValueError(
                "Falsification comparisons require a frozen falsification registry"
            )
        falsification_comparisons_bundle = load_falsification_comparisons(
            Path(falsification_comparisons_path),
            registry=falsification_bundle[0],
        )
    if not final_protocol:
        _validate_embedding_lock(cfg, required=False)
        _validate_judge_lock(cfg, required=False)

    embedding_views = cfg.get("text_embedding_views", [])
    if isinstance(embedding_views, str):
        embedding_views = [
            value.strip() for value in embedding_views.split(",") if value.strip()
        ]
    if embedding_views and (
        not set(embedding_views).issubset(ALLOWED_TEXT_VIEWS)
        or len(set(embedding_views)) != len(embedding_views)
    ):
        raise ValueError(
            f"Invalid text_embedding_views; choose from {sorted(ALLOWED_TEXT_VIEWS)}"
        )
    judge_views = cfg.get("llm_judge_views", [])
    if isinstance(judge_views, str):
        judge_views = [
            value.strip() for value in judge_views.split(",") if value.strip()
        ]
    if judge_views and (
        not set(judge_views).issubset(ALLOWED_TEXT_VIEWS)
        or len(set(judge_views)) != len(judge_views)
    ):
        raise ValueError(
            f"Invalid llm_judge_views; choose from {sorted(ALLOWED_TEXT_VIEWS)}"
        )
    judge_modes = cfg.get("llm_judge_modes", [])
    if isinstance(judge_modes, str):
        judge_modes = [
            value.strip() for value in judge_modes.split(",") if value.strip()
        ]
    if judge_modes and (
        not set(judge_modes).issubset({"zero_shot", "few_shot"})
        or len(set(judge_modes)) != len(judge_modes)
    ):
        raise ValueError("llm_judge_modes must contain zero_shot and/or few_shot")
    if int(cfg.get("llm_judge_batch_size", 8)) < 1:
        raise ValueError("llm_judge_batch_size must be positive")
    if cfg.get("run_black_box_baselines", False):
        _validate_embedding_lock(cfg, required=True)
        _validate_judge_lock(cfg, required=True)
        missing_baselines = REQUIRED_BLACK_BOX_BASELINES.difference(
            set(cfg.get("black_box_baselines", []))
        )
        if missing_baselines:
            raise ValueError(
                f"Configured black-box execution lacks baselines {sorted(missing_baselines)}"
            )
        if not embedding_views:
            raise ValueError(
                "Configured black-box execution requires text_embedding_views"
            )
        if not judge_views or set(judge_modes) != {"zero_shot", "few_shot"}:
            raise ValueError(
                "Configured black-box execution requires llm_judge_views and both judge modes"
            )

    models = cfg.get("models") or []
    if not models:
        raise ValueError("Config must define at least one model")
    falsification_manifests_by_model: dict[str, dict[str, dict]] = {}
    for model_cfg in models:
        for key in REQUIRED_MODEL_KEYS:
            if key not in model_cfg:
                raise ValueError(f"Model config missing key: {key}")
        revision = str(model_cfg["model_revision"])
        if not COMMIT_RE.fullmatch(revision):
            raise ValueError(
                f"Model {model_cfg['name']} must use a pinned commit-like model_revision"
            )
        if cfg.get("run_black_box_baselines", False):
            for key in (
                "labeled_data",
                "benign_labeled_data",
                "text_embedding_cache_dirs",
                "benign_embedding_cache_dir",
                "llm_judge_cache_dir",
            ):
                if not model_cfg.get(key):
                    raise ValueError(
                        f"Model {model_cfg['name']} must define {key} when black-box baselines run"
                    )
            missing_labeled = configured_tasks.difference(model_cfg["labeled_data"])
            missing_embedding = configured_tasks.difference(
                model_cfg["text_embedding_cache_dirs"]
            )
            if missing_labeled or missing_embedding:
                raise ValueError(
                    f"Model {model_cfg['name']} lacks black-box inputs for "
                    f"labeled={sorted(missing_labeled)}, embeddings={sorted(missing_embedding)}"
                )
        if run_falsification:
            if not model_cfg.get("labeled_data"):
                raise ValueError(
                    f"Model {model_cfg['name']} requires labeled_data for falsification"
                )
            if falsification_bundle is None:
                raise ValueError("Falsification execution requires a frozen registry")
            falsification_manifests_by_model[model_cfg["name"]] = (
                _validate_model_falsification_manifests(
                    model_cfg,
                    configured_tasks=configured_tasks,
                    registry_bundle=falsification_bundle,
                    check_paths=check_paths,
                    final_protocol=final_protocol,
                )
            )
        if check_paths:
            calibration_dirs = model_cfg.get("calibration_dirs", {})
            feature_groups: dict[str, dict[str, set[str]]] = {}
            for task_name, path_str in model_cfg["feature_dirs"].items():
                if (
                    task_name in TASK_REGISTRY
                    and TASK_REGISTRY[task_name]().spec.evaluation_only
                ):
                    required_splits = ("test",)
                else:
                    required_splits = (
                        ("train", "eval", "test")
                        if task_name in calibration_dirs
                        else ("train", "calibration", "eval", "test")
                    )
                feature_groups[task_name] = _validate_feature_directory(
                    Path(path_str),
                    model_cfg,
                    task_name,
                    int(cfg["min_calibration_negatives"]),
                    required_splits,
                    final_protocol,
                )
            for task_name, path_str in calibration_dirs.items():
                calibration_groups = _validate_feature_directory(
                    Path(path_str),
                    model_cfg,
                    f"{task_name}_benign_calibration",
                    int(cfg["min_calibration_negatives"]),
                    ("calibration",),
                    final_protocol,
                )
                source_groups = set().union(*feature_groups.get(task_name, {}).values())
                overlap = source_groups.intersection(
                    calibration_groups.get("calibration", set())
                )
                if overlap:
                    raise ValueError(
                        f"Dedicated calibration for {model_cfg['name']} / {task_name} overlaps "
                        f"{len(overlap)} source train/eval/test groups"
                    )
            embedding_dirs = model_cfg.get("text_embedding_cache_dirs", {})
            benign_embedding_dir = model_cfg.get("benign_embedding_cache_dir")
            if embedding_dirs or benign_embedding_dir:
                if not embedding_views:
                    raise ValueError(
                        "text_embedding_views are required to validate embedding caches"
                    )
                embedding_caches: list[dict] = []
                prompt_hashes_by_task: dict[str, set[str]] = {}
                for task_name, path_str in embedding_dirs.items():
                    task_caches = _validate_embedding_cache_directory(
                        Path(path_str),
                        task_name=task_name,
                        views=embedding_views,
                        model_cfg=model_cfg,
                        min_calibration_negatives=int(cfg["min_calibration_negatives"]),
                        calibration_only=False,
                    )
                    embedding_caches.extend(task_caches)
                    prompt_cache = next(
                        (
                            cache
                            for cache in task_caches
                            if cache["metadata"]["view"] == "prompt_text"
                        ),
                        None,
                    )
                    if prompt_cache is not None:
                        prompt_hashes_by_task[task_name] = set(
                            prompt_cache["normalized_text_sha256"].tolist()
                        )
                if benign_embedding_dir:
                    benign_caches = _validate_embedding_cache_directory(
                        Path(benign_embedding_dir),
                        task_name="benign_calibration",
                        views=embedding_views,
                        model_cfg=model_cfg,
                        min_calibration_negatives=int(cfg["min_calibration_negatives"]),
                        calibration_only=True,
                    )
                    embedding_caches.extend(benign_caches)
                    benign_prompt_cache = next(
                        (
                            cache
                            for cache in benign_caches
                            if cache["metadata"]["view"] == "prompt_text"
                        ),
                        None,
                    )
                    if benign_prompt_cache is not None:
                        prompt_hashes_by_task["benign_calibration"] = set(
                            benign_prompt_cache["normalized_text_sha256"].tolist()
                        )
                signatures = {_embedding_signature(cache) for cache in embedding_caches}
                if len(signatures) > 1:
                    raise ValueError(
                        f"Text embedding caches for {model_cfg['name']} do not share one frozen encoder spec"
                    )
                task_names = sorted(prompt_hashes_by_task)
                for index, task_a in enumerate(task_names):
                    for task_b in task_names[index + 1 :]:
                        overlap = prompt_hashes_by_task[task_a].intersection(
                            prompt_hashes_by_task[task_b]
                        )
                        if overlap:
                            raise ValueError(
                                f"Text embedding caches for {model_cfg['name']} contain "
                                f"{len(overlap)} normalized prompt overlaps between "
                                f"{task_a} and {task_b}"
                            )
            if cfg.get("run_black_box_baselines", False):
                for task_name, path_str in model_cfg["labeled_data"].items():
                    _validate_labeled_confidence_data(
                        Path(path_str),
                        task_name=task_name,
                        model_cfg=model_cfg,
                        calibration_only=False,
                        falsification_bundle=falsification_bundle,
                    )
                _validate_labeled_confidence_data(
                    Path(model_cfg["benign_labeled_data"]),
                    task_name="benign_calibration",
                    model_cfg=model_cfg,
                    calibration_only=True,
                    falsification_bundle=None,
                )

    if not final_protocol:
        _validate_benign_audit_protocol(cfg, required=False)

    if final_protocol:
        if cfg["protocol_stage"] != "frozen":
            raise ValueError("Final execution requires protocol_stage=frozen")
        if int(cfg["min_calibration_negatives"]) < 10_000:
            raise ValueError(
                "Final execution requires at least 10,000 independent benign calibration groups"
            )
        if int(cfg.get("seeds", 0)) < 10:
            raise ValueError(
                "Final execution requires at least 10 few-shot training seeds"
            )
        final_k_values = _parse_positive_int_set(
            cfg.get("k_values", ""), field="k_values"
        )
        if final_k_values != FINAL_K_VALUES:
            raise ValueError("Final execution requires k_values=1,2,4,8,16,32")
        if cfg.get("balance_modes") != "balanced":
            raise ValueError(
                "Final primary protocol must use matched balanced sampling only"
            )
        families = {str(model["family"]) for model in models}
        if len(families) < 3:
            raise ValueError(
                "Final execution requires at least three genuinely different model families"
            )
        _validate_benign_audit_protocol(cfg, required=True)
        _validate_embedding_lock(cfg, required=True)
        judge_model = _validate_judge_lock(cfg, required=True)
        if judge_model is None or judge_model.get("protocol_role") != "frozen_primary":
            raise ValueError("Final execution requires a frozen_primary LLM judge")
        registered_baselines = set(cfg.get("black_box_baselines", []))
        missing_baselines = sorted(
            REQUIRED_BLACK_BOX_BASELINES.difference(registered_baselines)
        )
        if missing_baselines:
            raise ValueError(
                f"Final execution lacks required black-box baselines: {missing_baselines}"
            )
        if set(embedding_views) != ALLOWED_TEXT_VIEWS:
            raise ValueError(
                "Final execution requires prompt_text, answer_text, and transcript_text embedding views"
            )
        if set(judge_views) != ALLOWED_TEXT_VIEWS:
            raise ValueError(
                "Final execution requires prompt_text, answer_text, and transcript_text judge views"
            )
        if set(judge_modes) != {"zero_shot", "few_shot"}:
            raise ValueError(
                "Final execution requires zero-shot and few-shot LLM judging"
            )
        if cfg.get("run_black_box_baselines") is not True:
            raise ValueError("Final execution must set run_black_box_baselines=true")
        if not run_falsification or falsification_bundle is None:
            raise ValueError("Final execution must set run_falsification_suite=true")
        if falsification_comparisons_bundle is None:
            raise ValueError(
                "Final execution requires an existing pre-registered falsification_comparisons_file"
            )
        source_tasks = {
            pair["source_task"]
            for key in ("task_pairs", "calibration_pairs", "transfer_pairs")
            for pair in cfg.get(key, [])
        }
        registered_tasks = configured_tasks
        for model in models:
            assert judge_model is not None
            if judge_model.get("model_id") == model["model_id"]:
                raise ValueError(
                    f"Final model {model['name']} cannot use itself as the independent LLM judge"
                )
            if (
                str(judge_model.get("family", "")).casefold()
                == str(model.get("family", "")).casefold()
            ):
                raise ValueError(
                    f"Final model {model['name']} must use a judge from a different model family"
                )
            missing_calibration = sorted(
                source_tasks.difference(model.get("calibration_dirs", {}))
            )
            if missing_calibration:
                raise ValueError(
                    f"Final model {model['name']} lacks dedicated benign calibration_dirs for {missing_calibration}"
                )
            missing_embedding_tasks = sorted(
                registered_tasks.difference(model.get("text_embedding_cache_dirs", {}))
            )
            if missing_embedding_tasks or not model.get("benign_embedding_cache_dir"):
                raise ValueError(
                    f"Final model {model['name']} lacks frozen embedding caches for "
                    f"tasks={missing_embedding_tasks} or dedicated benign calibration"
                )
            missing_labeled_tasks = sorted(
                registered_tasks.difference(model.get("labeled_data", {}))
            )
            if missing_labeled_tasks or not model.get("benign_labeled_data"):
                raise ValueError(
                    f"Final model {model['name']} lacks labeled text data for "
                    f"tasks={missing_labeled_tasks} or dedicated benign calibration"
                )
        _validate_final_falsification_coverage(
            cfg,
            models=models,
            manifests_by_model=falsification_manifests_by_model,
            registry=falsification_bundle[0],
            configured_tasks=configured_tasks,
        )
        _validate_final_falsification_comparison_coverage(
            falsification_comparisons_bundle[0],
            registry=falsification_bundle[0],
            configured_tasks=configured_tasks,
            configured_models={str(model["name"]) for model in models},
        )
        comparisons_file = cfg.get("comparisons_file")
        if not comparisons_file or not Path(comparisons_file).exists():
            raise ValueError(
                "Final execution requires an existing pre-registered comparisons_file"
            )
        comparison_cfg = (
            yaml.safe_load(Path(comparisons_file).read_text(encoding="utf-8")) or {}
        )
        comparisons = comparison_cfg.get("comparisons", [])
        if not comparisons:
            raise ValueError(
                "Final comparisons_file must define at least one comparison"
            )
        primary_comparisons = [
            comparison
            for comparison in comparisons
            if comparison.get("comparison_role") == "primary_white_box_gain"
        ]
        if not primary_comparisons:
            raise ValueError(
                "Final comparisons_file must define a primary_white_box_gain comparison"
            )
        for comparison in comparisons:
            for key in (
                "comparison_id",
                "description",
                "common_filters",
                "system_a",
                "system_b",
                "split",
                "metric",
            ):
                if key not in comparison:
                    raise ValueError(f"Pre-registered comparison is missing {key}")
            if "REPLACE" in str(comparison):
                raise ValueError(
                    "Final comparisons_file still contains placeholder values"
                )
            if comparison.get("comparison_role") == "primary_white_box_gain":
                if not str(comparison["system_a"].get("probe", "")).startswith("P"):
                    raise ValueError(
                        "Primary white-box gain system_a must be a registered P* monitor"
                    )
                if not str(comparison["system_b"].get("probe", "")).startswith("B"):
                    raise ValueError(
                        "Primary white-box gain system_b must be a registered B* monitor"
                    )
                missing_common = sorted(
                    {"model", "source_task", "target_task", "k"}.difference(
                        comparison["common_filters"]
                    )
                )
                if missing_common:
                    raise ValueError(
                        "Primary white-box gain comparison lacks common filters "
                        f"{missing_common}"
                    )
                for system_key in ("system_a", "system_b"):
                    missing_identity = sorted(
                        {"probe", "balance_mode", "layer", "view"}.difference(
                            comparison[system_key]
                        )
                    )
                    if missing_identity:
                        raise ValueError(
                            f"Primary {system_key} lacks exact identity fields "
                            f"{missing_identity}"
                        )
                if (
                    comparison["split"] != "target_test"
                    or comparison["metric"] != "tpr"
                ):
                    raise ValueError(
                        "Primary white-box gain must compare target_test TPR"
                    )

        configured_pairs = {
            (str(pair["source_task"]), str(pair["target_task"]))
            for key in ("task_pairs", "calibration_pairs", "transfer_pairs")
            for pair in cfg.get(key, [])
        }
        expected_primary = {
            (str(model["name"]), source_task, target_task, k)
            for model in models
            for source_task, target_task in configured_pairs
            for k in final_k_values
        }
        observed_primary: list[tuple[str, str, str, int]] = []
        for comparison in primary_comparisons:
            common = comparison["common_filters"]
            try:
                observed_primary.append(
                    (
                        str(common["model"]),
                        str(common["source_task"]),
                        str(common["target_task"]),
                        int(common["k"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as err:
                raise ValueError(
                    "Primary comparison has invalid model/source/target/k filters"
                ) from err
        if len(observed_primary) != len(set(observed_primary)):
            raise ValueError("Final comparisons_file duplicates a primary comparison")
        observed_primary_set = set(observed_primary)
        if observed_primary_set != expected_primary:
            missing_primary = sorted(expected_primary.difference(observed_primary_set))
            extra_primary = sorted(observed_primary_set.difference(expected_primary))
            raise ValueError(
                "Final comparisons_file has incomplete primary coverage: "
                f"missing={missing_primary[:5]}, extra={extra_primary[:5]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a leakage-resistant monitoring protocol"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--check_paths", action="store_true")
    parser.add_argument("--final_protocol", action="store_true")
    args = parser.parse_args()
    validate_config(
        load_yaml(args.config),
        check_paths=args.check_paths,
        final_protocol=args.final_protocol,
    )
    print("config validation passed")


if __name__ == "__main__":
    main()
