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
    "execution_mode",
    "results_dir",
    "models",
    "views",
    "layers",
    "probes",
    "k_values",
    "max_reference_alert_rate",
    "min_reference_groups",
    "selection_k",
    "seeds",
    "balance_modes",
    "bootstrap_samples",
    "run_black_box_baselines",
    "run_falsification_suite",
]
REQUIRED_MODEL_KEYS = [
    "name",
    "model_id",
    "model_revision",
    "family",
    "feature_dirs",
    "reference_feature_dir",
]
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
    if model.get("protocol_role") not in {"frozen_primary", "smoke_test_only"}:
        raise ValueError("LLM-judge model lock has invalid protocol_role")
    return model


def _validate_reference_protocol(cfg: dict, *, required: bool) -> None:
    protocol = cfg.get("reference_traffic_protocol")
    if protocol is None and not required:
        return
    if not isinstance(protocol, dict):
        raise ValueError("reference_traffic_protocol must be a mapping")
    if protocol.get("protocol") != "reference-traffic-v1":
        raise ValueError(
            "reference_traffic_protocol must freeze reference-traffic-v1"
        )
    if protocol.get("scope") != "per_monitored_model_revision":
        raise ValueError(
            "Reference traffic must be generated per monitored model revision"
        )
    if protocol.get("calibration_method") != "split_conformal_upper_tail_v1":
        raise ValueError(
            "Reference traffic must use split_conformal_upper_tail_v1"
        )
    maximum_rate = float(protocol.get("max_reference_alert_rate", 1.0))
    if not np.isclose(maximum_rate, float(cfg["max_reference_alert_rate"])):
        raise ValueError(
            "Reference protocol and top-level alert rates must be identical"
        )
    if not 0.0 < maximum_rate <= 0.01:
        raise ValueError("Reference alert budget cannot exceed 1%")
    if not isinstance(protocol.get("selection_seed"), int):
        raise ValueError("Reference sampling requires an integer selection_seed")
    calibration_groups = int(protocol.get("calibration_groups", 0))
    holdout_groups = int(protocol.get("holdout_groups", 0))
    if calibration_groups < int(cfg["min_reference_groups"]):
        raise ValueError(
            "Reference protocol calibration_groups is below min_reference_groups"
        )
    if holdout_groups < 1:
        raise ValueError("Reference protocol requires an untouched holdout")
    if protocol.get("semantic_labels") is not False:
        raise ValueError("Reference traffic must explicitly disable semantic labels")


def _validate_claim_gates(cfg: dict, *, required: bool) -> None:
    gates = cfg.get("claim_gates")
    if gates is None and not required:
        return
    if not isinstance(gates, dict):
        raise ValueError("claim_gates must be a mapping")
    expected = {
        "primary_effect": "white_box_minus_selected_black_box_tpr",
        "reference_holdout_check": "selected_system_interval_not_above_budget_v1",
        "hard_negative_check": "fpr_noninferiority_and_pairwise_ordering_v1",
        "multiplicity_control": "holm_global",
    }
    for key, value in expected.items():
        if gates.get(key) != value:
            raise ValueError(f"claim_gates.{key} must be {value!r}")
    margin = float(gates.get("hard_negative_fpr_noninferiority_margin", -1.0))
    if not 0.0 <= margin <= 0.05:
        raise ValueError(
            "hard_negative_fpr_noninferiority_margin must be in [0, 0.05]"
        )
    ordering_floor = float(gates.get("pairwise_order_accuracy_floor", -1.0))
    if not np.isclose(ordering_floor, 0.5):
        raise ValueError("pairwise_order_accuracy_floor must be 0.5")


def _validate_frozen_artifact_provenance(
    cfg: dict,
    *,
    registry_sha256: str,
    comparisons_path: Path,
    comparisons_config: dict,
    falsification_comparisons_path: Path,
    falsification_comparisons_config: dict,
) -> None:
    provenance = cfg.get("selection_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Frozen protocol requires selection_provenance")
    required_provenance = {
        "selection_file",
        "selection_file_sha256",
        "base_config_sha256",
        "selection_k",
        "registry_sha256",
    }
    missing = sorted(required_provenance.difference(provenance))
    if missing:
        raise ValueError(f"Frozen selection_provenance lacks fields {missing}")
    for key in (
        "selection_file_sha256",
        "base_config_sha256",
        "registry_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(provenance[key])):
            raise ValueError(f"Frozen selection_provenance has an invalid {key}")
    if int(provenance["selection_k"]) != int(cfg["selection_k"]):
        raise ValueError(
            "Frozen selection provenance disagrees with the registered selection_k"
        )
    if provenance["registry_sha256"] != registry_sha256:
        raise ValueError(
            "Frozen selection provenance disagrees with the falsification registry"
        )
    for label, artifact in (
        ("primary comparisons", comparisons_config),
        ("falsification comparisons", falsification_comparisons_config),
    ):
        if artifact.get("selection_provenance") != provenance:
            raise ValueError(
                f"Frozen {label} selection provenance disagrees with the manifest"
            )

    registered_hashes = cfg.get("registered_artifact_sha256")
    if not isinstance(registered_hashes, dict):
        raise ValueError("Frozen protocol requires registered_artifact_sha256")
    expected_hashes = {
        "comparisons_file": file_sha256(comparisons_path),
        "falsification_comparisons_file": file_sha256(
            falsification_comparisons_path
        ),
    }
    if registered_hashes != expected_hashes:
        raise ValueError(
            "Frozen registered comparison artifacts do not match their manifest hashes"
        )


def _validate_labeled_confidence_data(
    path: Path,
    *,
    task_name: str,
    model_cfg: dict,
    reference_only: bool,
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
    allowed_labels = set(task.spec.label_semantics)
    observed_labels = {int(example.label) for example in examples}
    if not observed_labels.issubset(allowed_labels):
        raise ValueError(
            f"Labeled data {path} contains labels outside the task contract: "
            f"{sorted(observed_labels - allowed_labels)}"
        )
    for example in examples:
        if confidence_supported:
            confidence_feature_vector(example.metadata.get("generation"))
        if falsification_bundle is not None and not reference_only:
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
    if reference_only:
        if not {"calibration", "test"}.issubset(observed_splits) or any(
            example.label != 0 for example in examples
        ):
            raise ValueError(
                f"Reference data {path} must contain membership-only calibration/test partitions"
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
    if not reference_only:
        required_splits = (
            ("test",)
            if task.spec.evaluation_only
            else ("train", "eval", "test")
        )
        for split in required_splits:
            split_labels = {
                int(example.label)
                for example in examples
                if example.metadata.get("protocol_split") == split
            }
            if split_labels != {0, 1}:
                raise ValueError(
                    f"Labeled confidence data {path} split {split} lacks both "
                    "objective behavior labels"
                )
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
    min_reference_groups: int,
    reference_only: bool,
    expected_dataset_sha256: str | None = None,
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
        if (
            expected_dataset_sha256 is not None
            and metadata["dataset_sha256"] != expected_dataset_sha256
        ):
            raise ValueError(
                f"Text embedding cache {cache_path} was built from a different "
                "labeled dataset"
            )
        if np.any(cache["truncated"]):
            raise ValueError(
                f"Final text embedding cache {cache_path} contains truncated inputs"
            )
        if reference_only:
            observed_splits = set(np.unique(cache["protocol_splits"]).tolist())
            if not {"calibration", "test"}.issubset(observed_splits):
                raise ValueError(
                    f"Reference cache {cache_path} requires calibration/test partitions"
                )
            if np.any(cache["labels"] != 0):
                raise ValueError(
                    f"Reference cache {cache_path} must use membership value 0"
                )
            calibration_mask = cache["protocol_splits"].astype(str) == "calibration"
            holdout_mask = cache["protocol_splits"].astype(str) == "test"
            if int(np.sum(calibration_mask)) < min_reference_groups:
                raise ValueError(
                    f"Reference cache {cache_path} has fewer than "
                    f"{min_reference_groups} calibration rows"
                )
            calibration_groups = np.unique(cache["question_ids"][calibration_mask])
            holdout_groups = np.unique(cache["question_ids"][holdout_mask])
            if len(calibration_groups) != int(np.sum(calibration_mask)):
                raise ValueError(
                    f"Reference cache {cache_path} repeats calibration groups"
                )
            if len(holdout_groups) != int(np.sum(holdout_mask)):
                raise ValueError(
                    f"Reference cache {cache_path} repeats holdout groups"
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
    min_reference_groups: int,
    required_splits: tuple[str, ...],
    final_protocol: bool,
    required_views: tuple[str, ...] = (),
    expected_dataset_sha256: str | None = None,
) -> dict[str, set[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature directory for {model_cfg['name']} / {task_name}: {path}"
        )
    representative_groups: dict[str, set[str]] = {}
    split_signatures: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    provenance_signatures: set[tuple[str, ...]] = set()
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
                    "tokenizer_revision",
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
                if str(bundle["tokenizer_revision"].item()) != model_cfg[
                    "model_revision"
                ]:
                    raise ValueError(
                        f"Feature bundle {bundle_path} uses a tokenizer revision "
                        "different from the registered monitored-model revision"
                    )
                if (
                    expected_dataset_sha256 is not None
                    and str(bundle["dataset_sha256"].item())
                    != expected_dataset_sha256
                ):
                    raise ValueError(
                        f"Feature bundle {bundle_path} was extracted from a "
                        "different labeled dataset"
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
                provenance_signatures.add(
                    tuple(
                        str(bundle[key].item())
                        for key in (
                            "dataset_sha256",
                            "code_revision",
                            "chat_template_sha256",
                            "extraction_config_sha256",
                            "model_revision",
                            "tokenizer_revision",
                        )
                    )
                )
                labels = np.asarray(bundle["labels"], dtype=int)
                if not set(np.unique(labels)).issubset({0, 1}):
                    raise ValueError(
                        f"Feature bundle {bundle_path} contains non-binary labels"
                    )
                missing_views = sorted(
                    set(required_views).difference(bundle.files)
                )
                if missing_views:
                    raise ValueError(
                        f"Feature bundle {bundle_path} lacks requested activation "
                        f"views {missing_views}"
                    )
                for view in required_views:
                    activations = np.asarray(bundle[view])
                    if (
                        activations.ndim != 2
                        or len(activations) != len(labels)
                        or not np.issubdtype(activations.dtype, np.number)
                        or not np.all(np.isfinite(activations))
                    ):
                        raise ValueError(
                            f"Feature bundle {bundle_path} has an invalid requested "
                            f"activation view {view!r}"
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
                    and int(np.sum(labels == 0)) < min_reference_groups
                ):
                    raise ValueError(
                        f"Feature bundle {bundle_path} has fewer than {min_reference_groups} calibration reference rows"
                    )
                if split == "calibration":
                    n_negative_groups = len(np.unique(question_ids[labels == 0]))
                    if n_negative_groups != int(np.sum(labels == 0)):
                        raise ValueError(
                            f"Feature bundle {bundle_path} repeats negative calibration groups"
                        )
                    if n_negative_groups < min_reference_groups:
                        raise ValueError(
                            f"Feature bundle {bundle_path} has only {n_negative_groups} independent "
                            f"reference calibration groups; requires {min_reference_groups}"
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
    if len(provenance_signatures) != 1:
        raise ValueError(
            f"Feature directory {path} mixes incompatible extraction provenance"
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
    required_axes = set(registry["required_final_heldout_axes"])
    covered_model_axes = {
        (
            str(row["common_filters"]["model"]),
            str(row["slice"]["axis"]),
        )
        for row in heldout_shift_comparisons
    }
    expected_model_axes = {
        (model, axis) for model in configured_models for axis in required_axes
    }
    if covered_model_axes != expected_model_axes:
        raise ValueError(
            "Final pre-registered falsification comparisons have incomplete "
            "model-by-heldout-axis coverage; missing "
            f"{sorted(expected_model_axes.difference(covered_model_axes))[:10]}"
        )

    heldout_behaviors = set(
        registry["behavior_transfer"]["heldout_values"]
    ).intersection(configured_tasks)
    covered_model_behavior_tasks = {
        (str(row["common_filters"]["model"]), str(row["task_name"]))
        for row in heldout_shift_comparisons
        if row["slice"]["axis"] == "behavior"
    }
    expected_model_behavior_tasks = {
        (model, task)
        for model in configured_models
        for task in heldout_behaviors
    }
    if covered_model_behavior_tasks != expected_model_behavior_tasks:
        raise ValueError(
            "Final pre-registered comparisons have incomplete model-by-behavior "
            f"coverage; missing {sorted(expected_model_behavior_tasks.difference(covered_model_behavior_tasks))[:10]}"
        )

    enabled_hard_tasks = {
        task_name
        for task_name in configured_tasks
        if registry["tasks"][task_name]["hard_negative"]["enabled"]
    }
    hard_comparisons = [
        row for row in comparisons if row["slice"]["type"] == "matched_hard_negative"
    ]
    covered_model_hard_tasks = {
        (str(row["common_filters"]["model"]), str(row["task_name"]))
        for row in hard_comparisons
    }
    expected_model_hard_tasks = {
        (model, task)
        for model in configured_models
        for task in enabled_hard_tasks
    }
    if covered_model_hard_tasks != expected_model_hard_tasks:
        raise ValueError(
            "Final pre-registered comparisons have incomplete model-by-hard-negative "
            f"coverage; missing {sorted(expected_model_hard_tasks.difference(covered_model_hard_tasks))[:10]}"
        )
    if any(
        row["metric"] not in HARD_NEGATIVE_COMPARISON_METRICS
        for row in hard_comparisons
    ):
        raise ValueError("Final hard-negative comparison uses an unregistered metric")
    required_hard_metrics = {"hard_negative_fpr", "pairwise_order_accuracy"}
    missing_hard_metrics = {
        f"{model}:{task_name}": sorted(
            required_hard_metrics.difference(
                {
                    row["metric"]
                    for row in hard_comparisons
                    if row["task_name"] == task_name
                    and str(row["common_filters"]["model"]) == model
                }
            )
        )
        for model in configured_models
        for task_name in enabled_hard_tasks
    }
    missing_hard_metrics = {
        task_name: metrics
        for task_name, metrics in missing_hard_metrics.items()
        if metrics
    }
    if missing_hard_metrics:
        raise ValueError(
            "Final hard-negative comparisons require FPR and pairwise-ordering "
            f"evidence per task; missing {missing_hard_metrics}"
        )


def validate_config(cfg: dict, *, check_paths: bool, final_protocol: bool) -> None:
    for key in REQUIRED_TOP_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing top-level key: {key}")
    k_values = _parse_positive_int_set(cfg.get("k_values", ""), field="k_values")
    try:
        selection_k = int(cfg["selection_k"])
    except (TypeError, ValueError) as err:
        raise ValueError("selection_k must be an integer") from err
    if selection_k not in k_values:
        raise ValueError(
            f"selection_k={selection_k} must be present in k_values={sorted(k_values)}"
        )
    for field in ("seeds", "bootstrap_samples"):
        try:
            value = int(cfg[field])
        except (TypeError, ValueError) as err:
            raise ValueError(f"{field} must be an integer") from err
        if value < 1:
            raise ValueError(f"{field} must be positive")
    if not str(cfg["views"]).strip() or not str(cfg["layers"]).strip():
        raise ValueError("views and layers must be non-empty")
    activation_views = tuple(
        value.strip()
        for value in str(cfg["views"]).split(",")
        if value.strip()
    )
    if len(set(activation_views)) != len(activation_views):
        raise ValueError("views must not contain duplicate activation views")
    probes = {
        value.strip()
        for value in str(cfg["probes"]).split(",")
        if value.strip()
    }
    if not probes:
        raise ValueError("probes must contain at least one registered monitor")
    balance_modes = {
        value.strip()
        for value in str(cfg["balance_modes"]).split(",")
        if value.strip()
    }
    if not balance_modes or not balance_modes.issubset({"balanced", "imbalanced"}):
        raise ValueError(
            "balance_modes must contain balanced and/or imbalanced"
        )
    for field in ("run_black_box_baselines", "run_falsification_suite"):
        if not isinstance(cfg[field], bool):
            raise ValueError(f"{field} must be boolean")
    if cfg["protocol_stage"] not in {"selection", "frozen"}:
        raise ValueError("protocol_stage must be 'selection' or 'frozen'")
    if cfg["execution_mode"] not in {"selection", "confirmatory"}:
        raise ValueError("execution_mode must be 'selection' or 'confirmatory'")
    expected_mode = {
        "selection": "selection",
        "frozen": "confirmatory",
    }[cfg["protocol_stage"]]
    if cfg["execution_mode"] != expected_mode:
        raise ValueError(
            f"protocol_stage={cfg['protocol_stage']} requires "
            f"execution_mode={expected_mode}"
        )
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
    if not np.isclose(float(cfg["max_reference_alert_rate"]), 0.01):
        raise ValueError(
            "The registered operating budget must be max_reference_alert_rate=0.01"
        )
    if int(cfg["min_reference_groups"]) < 1000:
        raise ValueError(
            "At least 1,000 independent reference groups are required for selection"
        )
    _validate_reference_protocol(cfg, required=final_protocol)
    _validate_claim_gates(cfg, required=final_protocol)
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
                "reference_data",
                "text_embedding_cache_dirs",
                "reference_embedding_cache_dir",
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
            labeled_hashes = {
                task_name: file_sha256(Path(path_value))
                for task_name, path_value in (
                    model_cfg.get("labeled_data", {}) or {}
                ).items()
                if Path(path_value).exists()
            }
            reference_data_path = model_cfg.get("reference_data")
            reference_data_sha256 = (
                file_sha256(Path(reference_data_path))
                if reference_data_path
                and Path(reference_data_path).exists()
                else None
            )
            feature_groups: dict[str, dict[str, set[str]]] = {}
            for task_name, path_str in model_cfg["feature_dirs"].items():
                if (
                    task_name in TASK_REGISTRY
                    and TASK_REGISTRY[task_name]().spec.evaluation_only
                ):
                    required_splits = ("test",)
                else:
                    required_splits = ("train", "eval", "test")
                feature_groups[task_name] = _validate_feature_directory(
                    Path(path_str),
                    model_cfg,
                    task_name,
                    int(cfg["min_reference_groups"]),
                    required_splits,
                    final_protocol,
                    activation_views,
                    labeled_hashes.get(task_name),
                )
            reference_groups = _validate_feature_directory(
                Path(model_cfg["reference_feature_dir"]),
                model_cfg,
                "reference_traffic",
                int(cfg["min_reference_groups"]),
                ("calibration", "test"),
                final_protocol,
                activation_views,
                reference_data_sha256,
            )
            if (
                final_protocol
                and len(reference_groups.get("test", set())) < 10_000
            ):
                raise ValueError(
                    f"Final reference holdout for {model_cfg['name']} requires "
                    "at least 10,000 independent groups"
                )
            all_source_groups = set().union(
                *(
                    split_groups
                    for groups in feature_groups.values()
                    for split_groups in groups.values()
                )
            )
            for split in ("calibration", "test"):
                overlap = all_source_groups.intersection(
                    reference_groups.get(split, set())
                )
                if overlap:
                    raise ValueError(
                        f"Reference {split} for {model_cfg['name']} overlaps "
                        f"{len(overlap)} behavior groups"
                    )
            embedding_dirs = model_cfg.get("text_embedding_cache_dirs", {})
            reference_embedding_dir = model_cfg.get("reference_embedding_cache_dir")
            if embedding_dirs or reference_embedding_dir:
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
                        min_reference_groups=int(cfg["min_reference_groups"]),
                        reference_only=False,
                        expected_dataset_sha256=labeled_hashes.get(task_name),
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
                if reference_embedding_dir:
                    reference_caches = _validate_embedding_cache_directory(
                        Path(reference_embedding_dir),
                        task_name="reference_traffic",
                        views=embedding_views,
                        model_cfg=model_cfg,
                        min_reference_groups=int(cfg["min_reference_groups"]),
                        reference_only=True,
                        expected_dataset_sha256=reference_data_sha256,
                    )
                    embedding_caches.extend(reference_caches)
                    reference_prompt_cache = next(
                        (
                            cache
                            for cache in reference_caches
                            if cache["metadata"]["view"] == "prompt_text"
                        ),
                        None,
                    )
                    if reference_prompt_cache is not None:
                        prompt_hashes_by_task["reference_traffic"] = set(
                            reference_prompt_cache["normalized_text_sha256"].tolist()
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
                        reference_only=False,
                        falsification_bundle=falsification_bundle,
                    )
                _validate_labeled_confidence_data(
                    Path(model_cfg["reference_data"]),
                    task_name="reference_traffic",
                    model_cfg=model_cfg,
                    reference_only=True,
                    falsification_bundle=None,
                )

    if final_protocol:
        if cfg["protocol_stage"] != "frozen":
            raise ValueError("Final execution requires protocol_stage=frozen")
        if int(cfg["min_reference_groups"]) < 10_000:
            raise ValueError(
                "Final execution requires at least 10,000 independent reference groups"
            )
        if int(cfg["reference_traffic_protocol"].get("holdout_groups", 0)) < 10_000:
            raise ValueError(
                "Final execution requires at least 10,000 reference holdout groups"
            )
        if int(cfg.get("seeds", 0)) < 10:
            raise ValueError(
                "Final execution requires at least 10 few-shot training seeds"
            )
        final_k_values = k_values
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
        _validate_reference_protocol(cfg, required=True)
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
            if not model.get("reference_feature_dir"):
                raise ValueError(
                    f"Final model {model['name']} lacks reference features"
                )
            missing_embedding_tasks = sorted(
                registered_tasks.difference(model.get("text_embedding_cache_dirs", {}))
            )
            if missing_embedding_tasks or not model.get("reference_embedding_cache_dir"):
                raise ValueError(
                    f"Final model {model['name']} lacks frozen embedding caches for "
                    f"tasks={missing_embedding_tasks} or reference traffic"
                )
            missing_labeled_tasks = sorted(
                registered_tasks.difference(model.get("labeled_data", {}))
            )
            if missing_labeled_tasks or not model.get("reference_data"):
                raise ValueError(
                    f"Final model {model['name']} lacks labeled text data for "
                    f"tasks={missing_labeled_tasks} or reference traffic"
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
        assert falsification_comparisons_bundle is not None
        assert falsification_bundle is not None
        _validate_frozen_artifact_provenance(
            cfg,
            registry_sha256=falsification_bundle[1],
            comparisons_path=Path(comparisons_file),
            comparisons_config=comparison_cfg,
            falsification_comparisons_path=Path(
                cfg["falsification_comparisons_file"]
            ),
            falsification_comparisons_config=falsification_comparisons_bundle[0],
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
