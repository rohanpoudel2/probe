from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from data.rollout_schema import content_hash, validate_messages


REFERENCE_TASK_FAMILY = "reference_traffic"
REFERENCE_LABEL_SOURCE = "unlabeled_reference_traffic_membership"
REFERENCE_PROTOCOL = "reference-traffic-v1"
REFERENCE_ANNOTATION_SCHEMA = "reference-traffic-annotation-v1"


def reference_text_sha256(rollout: dict[str, Any]) -> str:
    """Bind reference membership to the exact on-policy interaction."""

    return content_hash(
        {
            "prompt_messages": rollout.get("prompt_messages"),
            "response_text": rollout.get("response_text"),
        }
    )


def validate_reference_rollout(row: dict[str, Any]) -> None:
    """Validate objective integrity properties without assigning a semantic label."""

    rollout_id = str(row.get("rollout_id", "")).strip()
    if not rollout_id:
        raise ValueError("Every reference rollout requires rollout_id")
    if row.get("task_family") != REFERENCE_TASK_FAMILY:
        raise ValueError(f"Rollout {rollout_id} is not reference traffic")
    if row.get("protocol_split") not in {"calibration", "test"}:
        raise ValueError(
            f"Reference rollout {rollout_id} must be in calibration or test"
        )
    if (
        row.get("data_origin") != "on_policy_generation"
        or row.get("generated_by_model") is not True
    ):
        raise ValueError(f"Reference rollout {rollout_id} is not on-policy")
    for key in ("model_id", "model_revision", "tokenizer_revision", "group_id"):
        if not str(row.get(key, "")).strip():
            raise ValueError(f"Reference rollout {rollout_id} lacks {key}")

    generation = row.get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"Reference rollout {rollout_id} lacks generation metadata")
    stop_reason = generation.get("stop_reason")
    response_token_count = generation.get("response_token_count")
    max_new_tokens = generation.get("max_new_tokens")
    hit_length_cap = (
        isinstance(response_token_count, int)
        and not isinstance(response_token_count, bool)
        and isinstance(max_new_tokens, int)
        and not isinstance(max_new_tokens, bool)
        and response_token_count >= max_new_tokens
    )
    if stop_reason == "max_new_tokens" or (stop_reason is None and hit_length_cap):
        raise ValueError(
            f"Reference rollout {rollout_id} hit max_new_tokens and is incomplete"
        )

    prompts = validate_messages(row.get("prompt_messages"), allow_assistant=False)
    messages = validate_messages(row.get("messages"), allow_assistant=True)
    if messages[:-1] != prompts or messages[-1]["content"] != row.get("response_text"):
        raise ValueError(
            f"Reference rollout {rollout_id} has inconsistent prompt/response messages"
        )


def build_reference_annotations(
    rollouts: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create reference-membership records after fail-closed integrity validation."""

    rows = list(rollouts)
    if not rows:
        raise ValueError("Reference traffic is empty")

    seen_rollouts: set[str] = set()
    seen_groups: set[str] = set()
    model_identities: set[tuple[str, str, str]] = set()
    split_counts: Counter[str] = Counter()
    annotations: list[dict[str, Any]] = []
    for row in rows:
        validate_reference_rollout(row)
        rollout_id = str(row["rollout_id"])
        group_id = str(row["group_id"])
        if rollout_id in seen_rollouts:
            raise ValueError(f"Duplicate reference rollout_id {rollout_id}")
        if group_id in seen_groups:
            raise ValueError(
                "Reference traffic requires exactly one rollout per independent "
                f"group; repeated group {group_id}"
            )
        seen_rollouts.add(rollout_id)
        seen_groups.add(group_id)
        model_identities.add(
            (
                str(row["model_id"]),
                str(row["model_revision"]),
                str(row["tokenizer_revision"]),
            )
        )
        split = str(row["protocol_split"])
        split_counts[split] += 1
        annotations.append(
            {
                "annotation_schema_version": REFERENCE_ANNOTATION_SCHEMA,
                "rollout_id": rollout_id,
                # This value encodes reference-set membership for shared feature
                # infrastructure. It is explicitly not a semantic negative label.
                "label": 0,
                "excluded": False,
                "label_source": REFERENCE_LABEL_SOURCE,
                "annotation_protocol": REFERENCE_PROTOCOL,
                "metadata": {
                    "reference_membership_only": True,
                    "semantic_negative_label": False,
                    "reference_partition": split,
                    "reference_text_sha256": reference_text_sha256(row),
                },
            }
        )

    if len(model_identities) != 1:
        raise ValueError(
            "Reference traffic must contain exactly one monitored model/tokenizer revision"
        )
    if set(split_counts) != {"calibration", "test"}:
        raise ValueError(
            "Reference traffic must contain both calibration and test partitions"
        )
    report = {
        "status": "pass",
        "protocol": REFERENCE_PROTOCOL,
        "n_reference_rows": len(rows),
        "n_independent_groups": len(seen_groups),
        "split_counts": dict(sorted(split_counts.items())),
        "model_identity": list(next(iter(model_identities))),
        "semantic_labels_assigned": False,
        "reference_membership_value": 0,
    }
    return annotations, report


def validate_reference_annotation_metadata(
    rollout: dict[str, Any], metadata: Any
) -> None:
    validate_reference_rollout(rollout)
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Reference rollout {rollout.get('rollout_id')} lacks metadata"
        )
    if metadata.get("reference_membership_only") is not True:
        raise ValueError("Reference membership must be explicit")
    if metadata.get("semantic_negative_label") is not False:
        raise ValueError("Reference traffic must not claim a semantic negative label")
    if metadata.get("reference_partition") != rollout.get("protocol_split"):
        raise ValueError("Reference partition does not match the rollout split")
    if metadata.get("reference_text_sha256") != reference_text_sha256(rollout):
        raise ValueError("Reference traffic annotation has a stale text hash")
