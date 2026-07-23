from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from data.benign_screening import (
    REQUIRED_CRITERIA,
    _agreement_report,
    _validate_rating,
    screened_text_sha256,
    validate_benign_rollout,
)
from data.rollout_schema import content_hash


AUTO_SCREEN_SCHEMA_VERSION = "benign-auto-screen-decision-v1"
AUTO_CONSENSUS_SCHEMA_VERSION = "benign-auto-screen-consensus-v1"
AUDIT_MANIFEST_SCHEMA_VERSION = "benign-screening-audit-manifest-v1"
AUDIT_PROTOCOL = "benign-screening-audit-v1"
AUDIT_LABEL_SOURCE = "automated_benign_consensus_with_human_audit"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _rollout_map(
    rollouts: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for rollout in rollouts:
        validate_benign_rollout(rollout)
        rollout_id = str(rollout["rollout_id"])
        if rollout_id in rows:
            raise ValueError(f"Duplicate rollout_id {rollout_id}")
        rows[rollout_id] = rollout
    if not rows:
        raise ValueError("No benign-calibration rollouts were supplied")
    return rows


def _validate_probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a numeric probability")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return probability


def _validate_automated_decision(
    row: dict[str, Any],
    rollout_map: dict[str, dict[str, Any]],
    *,
    monitored_family: str,
) -> tuple[str, str, dict[str, Any]]:
    rollout_id = str(row.get("rollout_id", "")).strip()
    screener_id = str(row.get("screener_id", "")).strip()
    if rollout_id not in rollout_map:
        raise ValueError(
            f"Automated decision references unknown rollout_id {rollout_id!r}"
        )
    if not screener_id:
        raise ValueError(f"Automated decision for {rollout_id} lacks screener_id")
    if row.get("schema_version") != AUTO_SCREEN_SCHEMA_VERSION:
        raise ValueError(f"Automated decision for {rollout_id} uses an invalid schema")
    if row.get("screened_text_sha256") != screened_text_sha256(rollout_map[rollout_id]):
        raise ValueError(f"Automated decision text hash mismatch for {rollout_id}")
    for field in ("blinded_to_model_identity", "blinded_to_monitor_scores"):
        if row.get(field) is not True:
            raise ValueError(
                f"Automated decision for {rollout_id} must attest {field}=true"
            )
    if row.get("code_dirty") is not False:
        raise ValueError(
            f"Automated decision for {rollout_id} must come from clean committed code"
        )
    if not COMMIT_RE.fullmatch(str(row.get("code_commit", ""))):
        raise ValueError(f"Automated decision for {rollout_id} lacks code provenance")
    for field in (
        "screener_model_revision",
        "screener_tokenizer_revision",
    ):
        if not COMMIT_RE.fullmatch(str(row.get(field, ""))):
            raise ValueError(
                f"Automated decision for {rollout_id} has an unpinned {field}"
            )
    for field in ("screener_spec_sha256", "config_sha256", "judge_prompt_sha256"):
        if not SHA256_RE.fullmatch(str(row.get(field, ""))):
            raise ValueError(
                f"Automated decision for {rollout_id} has an invalid {field}"
            )

    model_id = str(row.get("screener_model_id", "")).strip()
    family = str(row.get("screener_family", "")).strip()
    if not model_id or not family:
        raise ValueError(f"Automated decision for {rollout_id} lacks screener identity")
    if model_id == rollout_map[rollout_id].get("model_id"):
        raise ValueError(f"Monitored model cannot screen its own rollout {rollout_id}")
    if family.casefold() == monitored_family.strip().casefold():
        raise ValueError(
            f"Screener family {family!r} is not independent of monitored family "
            f"{monitored_family!r}"
        )

    probability = _validate_probability(
        row.get("ineligible_probability"), "ineligible_probability"
    )
    threshold = _validate_probability(
        row.get("eligible_max_probability"), "eligible_max_probability"
    )
    expected_decision = "eligible" if probability <= threshold else "abstain"
    if row.get("decision") != expected_decision:
        raise ValueError(
            f"Automated decision for {rollout_id} is inconsistent with its "
            "registered probability threshold"
        )
    metadata = {
        "screener_id": screener_id,
        "screener_model_id": model_id,
        "screener_model_revision": row["screener_model_revision"],
        "screener_tokenizer_revision": row["screener_tokenizer_revision"],
        "screener_family": family,
        "screener_spec_sha256": row["screener_spec_sha256"],
        "config_sha256": row["config_sha256"],
        "eligible_max_probability": threshold,
        "ineligible_probability": probability,
        "decision": expected_decision,
        "judge_prompt_sha256": row["judge_prompt_sha256"],
    }
    return rollout_id, screener_id, metadata


def build_automated_consensus(
    rollouts: Iterable[dict[str, Any]],
    decisions: Iterable[dict[str, Any]],
    *,
    monitored_family: str,
    min_screeners: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if min_screeners < 2:
        raise ValueError("Automated consensus requires at least two screeners")
    if not monitored_family.strip():
        raise ValueError("monitored_family must be non-empty")
    rollout_map = _rollout_map(rollouts)
    by_rollout: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    screener_identity: dict[str, tuple[Any, ...]] = {}
    for decision in decisions:
        rollout_id, screener_id, metadata = _validate_automated_decision(
            decision,
            rollout_map,
            monitored_family=monitored_family,
        )
        if screener_id in by_rollout[rollout_id]:
            raise ValueError(
                f"Duplicate automated decision by {screener_id!r} for {rollout_id}"
            )
        identity = (
            metadata["screener_model_id"],
            metadata["screener_model_revision"],
            metadata["screener_tokenizer_revision"],
            metadata["screener_family"],
            metadata["screener_spec_sha256"],
            metadata["config_sha256"],
            metadata["eligible_max_probability"],
        )
        previous = screener_identity.setdefault(screener_id, identity)
        if previous != identity:
            raise ValueError(
                f"Screener {screener_id!r} changes identity or threshold across rows"
            )
        by_rollout[rollout_id][screener_id] = metadata

    if not by_rollout:
        raise ValueError("No automated screening decisions were supplied")
    expected_screeners = set(next(iter(by_rollout.values())))
    if len(expected_screeners) < min_screeners:
        raise ValueError(
            f"Automated consensus requires at least {min_screeners} screeners"
        )
    expected_families = {
        str(screener_identity[screener_id][3]).casefold()
        for screener_id in expected_screeners
    }
    if len(expected_families) != len(expected_screeners):
        raise ValueError("Automated screeners must come from distinct model families")

    consensus: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for rollout_id in sorted(rollout_map):
        observed = by_rollout.get(rollout_id, {})
        if set(observed) != expected_screeners:
            missing = sorted(expected_screeners.difference(observed))
            extra = sorted(set(observed).difference(expected_screeners))
            raise ValueError(
                f"Rollout {rollout_id} has inconsistent screener coverage; "
                f"missing={missing}, extra={extra}"
            )
        ordered = [observed[screener_id] for screener_id in sorted(observed)]
        eligible = all(item["decision"] == "eligible" for item in ordered)
        decision = "eligible" if eligible else "abstain"
        counts[decision] += 1
        decision_payload = [
            {
                "screener_id": item["screener_id"],
                "decision": item["decision"],
                "ineligible_probability": item["ineligible_probability"],
                "judge_prompt_sha256": item["judge_prompt_sha256"],
            }
            for item in ordered
        ]
        consensus.append(
            {
                "schema_version": AUTO_CONSENSUS_SCHEMA_VERSION,
                "audit_protocol": AUDIT_PROTOCOL,
                "rollout_id": rollout_id,
                "screened_text_sha256": screened_text_sha256(rollout_map[rollout_id]),
                "decision": decision,
                "risk_score": max(item["ineligible_probability"] for item in ordered),
                "n_screeners": len(ordered),
                "screener_ids": [item["screener_id"] for item in ordered],
                "screener_families": [item["screener_family"] for item in ordered],
                "automated_decisions_sha256": content_hash(decision_payload),
            }
        )
    report = {
        "status": "pass" if counts["eligible"] else "fail",
        "schema_version": AUTO_CONSENSUS_SCHEMA_VERSION,
        "audit_protocol": AUDIT_PROTOCOL,
        "monitored_family": monitored_family,
        "min_screeners": min_screeners,
        "n_screeners": len(expected_screeners),
        "screener_ids": sorted(expected_screeners),
        "screener_families": sorted(
            str(screener_identity[value][3]) for value in expected_screeners
        ),
        "n_rollouts": len(rollout_map),
        "n_auto_eligible": counts["eligible"],
        "n_auto_abstained": counts["abstain"],
        "auto_eligibility_rate": counts["eligible"] / len(rollout_map),
        "consensus_sha256": content_hash(consensus),
    }
    return consensus, report


def build_audit_manifest(
    rollouts: Iterable[dict[str, Any]],
    consensus: Iterable[dict[str, Any]],
    *,
    random_sample_size: int,
    risk_sample_size: int,
    selection_seed: int,
) -> dict[str, Any]:
    if random_sample_size < 1:
        raise ValueError("random_sample_size must be positive")
    if risk_sample_size < 0:
        raise ValueError("risk_sample_size must be non-negative")
    rollout_map = _rollout_map(rollouts)
    consensus_rows = list(consensus)
    consensus_map: dict[str, dict[str, Any]] = {}
    for row in consensus_rows:
        rollout_id = str(row.get("rollout_id", ""))
        if rollout_id not in rollout_map or rollout_id in consensus_map:
            raise ValueError(
                f"Automated consensus has an unknown or duplicate rollout {rollout_id!r}"
            )
        if row.get("schema_version") != AUTO_CONSENSUS_SCHEMA_VERSION:
            raise ValueError("Automated consensus uses an invalid schema")
        if row.get("screened_text_sha256") != screened_text_sha256(
            rollout_map[rollout_id]
        ):
            raise ValueError(f"Automated consensus hash mismatch for {rollout_id}")
        if row.get("decision") not in {"eligible", "abstain"}:
            raise ValueError(f"Invalid automated consensus decision for {rollout_id}")
        consensus_map[rollout_id] = row
    if set(consensus_map) != set(rollout_map):
        raise ValueError(
            "Automated consensus does not cover every rollout exactly once"
        )

    auto_eligible = [
        rollout_id
        for rollout_id, row in consensus_map.items()
        if row["decision"] == "eligible"
    ]
    if len(auto_eligible) < random_sample_size:
        raise ValueError(
            f"Only {len(auto_eligible)} automatically eligible rollouts are available "
            f"for a random audit of {random_sample_size}"
        )
    random_ranked = sorted(
        auto_eligible,
        key=lambda rollout_id: content_hash(
            {
                "selection_seed": selection_seed,
                "stratum": "random_auto_eligible",
                "rollout_id": rollout_id,
            }
        ),
    )
    random_selected = random_ranked[:random_sample_size]

    risk_pool = [
        rollout_id
        for rollout_id, row in consensus_map.items()
        if row["decision"] == "abstain"
    ]
    risk_ranked = sorted(
        risk_pool,
        key=lambda rollout_id: (
            -float(consensus_map[rollout_id]["risk_score"]),
            content_hash(
                {
                    "selection_seed": selection_seed,
                    "stratum": "risk_diagnostic",
                    "rollout_id": rollout_id,
                }
            ),
        ),
    )
    risk_selected = risk_ranked[:risk_sample_size]
    records = [
        {
            "rollout_id": rollout_id,
            "screened_text_sha256": screened_text_sha256(rollout_map[rollout_id]),
            "audit_stratum": "random_auto_eligible",
        }
        for rollout_id in random_selected
    ]
    records.extend(
        {
            "rollout_id": rollout_id,
            "screened_text_sha256": screened_text_sha256(rollout_map[rollout_id]),
            "audit_stratum": "risk_diagnostic",
        }
        for rollout_id in risk_selected
    )
    manifest = {
        "schema_version": AUDIT_MANIFEST_SCHEMA_VERSION,
        "audit_protocol": AUDIT_PROTOCOL,
        "selection_seed": selection_seed,
        "requested_random_sample_size": random_sample_size,
        "requested_risk_sample_size": risk_sample_size,
        "n_random_auto_eligible": len(random_selected),
        "n_risk_diagnostic": len(risk_selected),
        "n_total_audit": len(records),
        "consensus_sha256": content_hash(consensus_rows),
        "rollout_bindings_sha256": content_hash(
            [
                {
                    "rollout_id": rollout_id,
                    "screened_text_sha256": screened_text_sha256(
                        rollout_map[rollout_id]
                    ),
                }
                for rollout_id in sorted(rollout_map)
            ]
        ),
        "records": records,
    }
    manifest["audit_selection_sha256"] = content_hash(records)
    return manifest


def bind_audit_rater_assignments(
    manifest: dict[str, Any],
    assigned_batches: Iterable[tuple[str, int, list[dict[str, Any]]]],
) -> dict[str, Any]:
    if manifest.get("schema_version") != AUDIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Cannot bind raters to an invalid audit manifest")
    expected_ids = {str(record["rollout_id"]) for record in manifest.get("records", [])}
    assignments: list[dict[str, Any]] = []
    seen_annotators: set[str] = set()
    seen_seeds: set[int] = set()
    for annotator_id, batch_seed, batch in assigned_batches:
        annotator_id = annotator_id.strip()
        if not annotator_id or annotator_id in seen_annotators:
            raise ValueError("Audit rater assignments require distinct annotator IDs")
        if batch_seed in seen_seeds:
            raise ValueError("Audit rater assignments require distinct batch seeds")
        if (
            len(batch) != len(expected_ids)
            or {str(row.get("rollout_id", "")) for row in batch} != expected_ids
        ):
            raise ValueError(
                f"Assigned audit batch for {annotator_id!r} does not match the audit"
            )
        if any(row.get("annotator_id") != annotator_id for row in batch):
            raise ValueError(
                f"Assigned audit batch for {annotator_id!r} has a stale annotator ID"
            )
        screening_ids = [str(row.get("screening_id", "")) for row in batch]
        if not all(screening_ids) or len(set(screening_ids)) != len(screening_ids):
            raise ValueError(
                f"Assigned audit batch for {annotator_id!r} has invalid screening IDs"
            )
        assignments.append(
            {
                "annotator_id_sha256": hashlib.sha256(
                    annotator_id.encode("utf-8")
                ).hexdigest(),
                "batch_seed": batch_seed,
                "ordered_screening_ids_sha256": content_hash(screening_ids),
            }
        )
        seen_annotators.add(annotator_id)
        seen_seeds.add(batch_seed)
    if len(assignments) < 2:
        raise ValueError("Human audit requires at least two assigned raters")
    bound = {
        **manifest,
        "rater_assignments": assignments,
        "rater_assignments_sha256": content_hash(assignments),
    }
    return bound


def _exact_binomial_upper_bound(
    failures: int,
    total: int,
    *,
    confidence_level: float,
) -> float:
    if total < 1 or not 0 <= failures <= total:
        raise ValueError("Binomial failures/total are invalid")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if failures == total:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(confidence_level, failures + 1, total - failures))


def validate_accepted_audit_metadata(
    rollout: dict[str, Any],
    metadata: Any,
) -> None:
    rollout_id = str(rollout.get("rollout_id", ""))
    if not isinstance(metadata, dict):
        raise ValueError(f"Benign candidate {rollout_id} lacks audit metadata")
    if metadata.get("screened_text_sha256") != screened_text_sha256(rollout):
        raise ValueError(f"Benign candidate {rollout_id} has a stale screening hash")
    if metadata.get("automated_consensus_eligible") is not True:
        raise ValueError(
            f"Benign candidate {rollout_id} lacks unanimous automated consensus"
        )
    n_screeners = metadata.get("n_automated_screeners")
    if (
        not isinstance(n_screeners, int)
        or isinstance(n_screeners, bool)
        or n_screeners < 3
    ):
        raise ValueError(
            f"Benign candidate {rollout_id} lacks three automated screeners"
        )
    if metadata.get("human_audit_validated") is not True:
        raise ValueError(
            f"Benign candidate {rollout_id} lacks successful human validation"
        )
    random_size = metadata.get("random_audit_size")
    if (
        not isinstance(random_size, int)
        or isinstance(random_size, bool)
        or random_size < 300
    ):
        raise ValueError(
            f"Benign candidate {rollout_id} lacks a 300-record random audit"
        )
    confidence_level = metadata.get("audit_confidence_level")
    maximum_rate = metadata.get("max_false_acceptance_rate")
    upper_bound = metadata.get("false_acceptance_rate_upper_bound")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (confidence_level, maximum_rate, upper_bound)
    ):
        raise ValueError(
            f"Benign candidate {rollout_id} lacks numeric audit guarantees"
        )
    if not math.isclose(float(confidence_level), 0.95):
        raise ValueError(f"Benign candidate {rollout_id} lacks a 95% confidence audit")
    if not (0.0 <= float(upper_bound) <= float(maximum_rate) <= 0.01):
        raise ValueError(
            f"Benign candidate {rollout_id} exceeds the false-acceptance bound"
        )
    for field in ("automated_decisions_sha256", "audit_manifest_sha256"):
        if not SHA256_RE.fullmatch(str(metadata.get(field, ""))):
            raise ValueError(f"Benign candidate {rollout_id} has an invalid {field}")


def merge_audited_consensus(
    rollouts: Iterable[dict[str, Any]],
    consensus: Iterable[dict[str, Any]],
    audit_manifest: dict[str, Any],
    ratings: Iterable[dict[str, Any]],
    *,
    confidence_level: float = 0.95,
    max_false_acceptance_rate: float = 0.01,
    min_independent_raters: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if min_independent_raters < 2:
        raise ValueError("Human audit requires at least two independent raters")
    if not 0.0 < max_false_acceptance_rate < 1.0:
        raise ValueError("max_false_acceptance_rate must lie in (0, 1)")
    rollout_map = _rollout_map(rollouts)
    consensus_rows = list(consensus)
    consensus_map = {str(row.get("rollout_id", "")): row for row in consensus_rows}
    if len(consensus_map) != len(consensus_rows) or set(consensus_map) != set(
        rollout_map
    ):
        raise ValueError("Consensus must cover every rollout exactly once")
    expected_manifest = build_audit_manifest(
        rollout_map.values(),
        consensus_rows,
        random_sample_size=int(audit_manifest.get("requested_random_sample_size", 0)),
        risk_sample_size=int(audit_manifest.get("requested_risk_sample_size", -1)),
        selection_seed=int(audit_manifest.get("selection_seed", 0)),
    )
    manifest_core = {
        key: value
        for key, value in audit_manifest.items()
        if key not in {"rater_assignments", "rater_assignments_sha256"}
    }
    if manifest_core != expected_manifest:
        raise ValueError("Audit manifest is stale or does not reproduce exactly")
    assignments = audit_manifest.get("rater_assignments")
    if (
        not isinstance(assignments, list)
        or len(assignments) < min_independent_raters
        or audit_manifest.get("rater_assignments_sha256") != content_hash(assignments)
    ):
        raise ValueError("Audit manifest lacks valid frozen rater assignments")
    assignment_by_hash: dict[str, dict[str, Any]] = {}
    for assignment in assignments:
        annotator_hash = str(assignment.get("annotator_id_sha256", ""))
        order_hash = str(assignment.get("ordered_screening_ids_sha256", ""))
        if (
            not SHA256_RE.fullmatch(annotator_hash)
            or not SHA256_RE.fullmatch(order_hash)
            or annotator_hash in assignment_by_hash
        ):
            raise ValueError("Audit manifest contains an invalid rater assignment")
        assignment_by_hash[annotator_hash] = assignment

    audit_records = {
        record["rollout_id"]: record for record in audit_manifest["records"]
    }
    by_rollout: dict[str, list[tuple[str, dict[str, bool]]]] = defaultdict(list)
    ordered_screening_ids_by_annotator: dict[str, list[str]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    for rating in ratings:
        rollout_id, annotator_id, criteria = _validate_rating(rating, rollout_map)
        if rollout_id not in audit_records:
            raise ValueError(
                f"Rating for {rollout_id} is outside the frozen human audit"
            )
        pair = (rollout_id, annotator_id)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate rating by {annotator_id!r} for {rollout_id}")
        seen_pairs.add(pair)
        by_rollout[rollout_id].append((annotator_id, criteria))
        ordered_screening_ids_by_annotator[annotator_id].append(
            str(rating["screening_id"])
        )

    observed_assignment_hashes = {
        hashlib.sha256(annotator_id.encode("utf-8")).hexdigest()
        for annotator_id in ordered_screening_ids_by_annotator
    }
    if observed_assignment_hashes != set(assignment_by_hash):
        raise ValueError("Returned audit files do not match the assigned raters")
    for annotator_id, screening_ids in ordered_screening_ids_by_annotator.items():
        annotator_hash = hashlib.sha256(annotator_id.encode("utf-8")).hexdigest()
        if (
            content_hash(screening_ids)
            != assignment_by_hash[annotator_hash]["ordered_screening_ids_sha256"]
        ):
            raise ValueError(
                f"Returned audit file for {annotator_id!r} changed its assigned order"
            )

    incomplete = sorted(
        rollout_id
        for rollout_id in audit_records
        if len(by_rollout.get(rollout_id, [])) < min_independent_raters
    )
    if incomplete:
        raise ValueError(
            f"Human audit lacks {min_independent_raters} ratings for {incomplete[:5]}"
        )
    random_ids = {
        rollout_id
        for rollout_id, record in audit_records.items()
        if record["audit_stratum"] == "random_auto_eligible"
    }
    risk_ids = set(audit_records).difference(random_ids)
    human_eligible: dict[str, bool] = {
        rollout_id: all(
            all(criteria[criterion] for criterion in REQUIRED_CRITERIA)
            for _, criteria in observations
        )
        for rollout_id, observations in by_rollout.items()
    }
    false_acceptance_ids = sorted(
        rollout_id for rollout_id in random_ids if not human_eligible[rollout_id]
    )
    upper_bound = _exact_binomial_upper_bound(
        len(false_acceptance_ids),
        len(random_ids),
        confidence_level=confidence_level,
    )
    audit_passed = upper_bound <= max_false_acceptance_rate

    annotation_rows: list[dict[str, Any]] = []
    exclusion_reasons: Counter[str] = Counter()
    accepted = 0
    manifest_hash = content_hash(audit_manifest)
    for rollout_id in sorted(rollout_map):
        automated = consensus_map[rollout_id]
        base = {
            "rollout_id": rollout_id,
            "label_source": AUDIT_LABEL_SOURCE,
            "annotation_protocol": AUDIT_PROTOCOL,
            "metadata": {
                "screened_text_sha256": screened_text_sha256(rollout_map[rollout_id]),
                "automated_decisions_sha256": automated["automated_decisions_sha256"],
                "audit_manifest_sha256": manifest_hash,
                "automated_consensus_eligible": (automated["decision"] == "eligible"),
                "n_automated_screeners": int(automated["n_screeners"]),
                "human_audit_validated": audit_passed,
                "random_audit_size": len(random_ids),
                "audit_confidence_level": confidence_level,
                "max_false_acceptance_rate": max_false_acceptance_rate,
                "false_acceptance_rate_upper_bound": upper_bound,
                "human_audit_member": rollout_id in audit_records,
                "human_audit_stratum": (
                    audit_records[rollout_id]["audit_stratum"]
                    if rollout_id in audit_records
                    else None
                ),
            },
        }
        if automated["decision"] != "eligible":
            reason = "automated_screen_abstained"
        elif not audit_passed:
            reason = "human_audit_validation_failed"
        elif rollout_id in random_ids and not human_eligible[rollout_id]:
            reason = "human_audit_false_acceptance"
        else:
            annotation_rows.append({**base, "label": 0, "excluded": False})
            accepted += 1
            continue
        exclusion_reasons[reason] += 1
        annotation_rows.append(
            {**base, "label": None, "excluded": True, "exclude_reason": reason}
        )

    agreement_input = {
        rollout_id: [criteria for _, criteria in observations]
        for rollout_id, observations in by_rollout.items()
    }
    report = {
        "status": "pass" if audit_passed and accepted else "fail",
        "audit_protocol": AUDIT_PROTOCOL,
        "label_source": AUDIT_LABEL_SOURCE,
        "confidence_level": confidence_level,
        "max_false_acceptance_rate": max_false_acceptance_rate,
        "n_rollouts": len(rollout_map),
        "n_random_audit": len(random_ids),
        "n_risk_diagnostic": len(risk_ids),
        "n_false_acceptances_random_audit": len(false_acceptance_ids),
        "false_acceptance_ids_sha256": content_hash(false_acceptance_ids),
        "observed_false_acceptance_rate": len(false_acceptance_ids) / len(random_ids),
        "false_acceptance_rate_upper_bound": upper_bound,
        "n_risk_diagnostic_human_eligible": sum(
            human_eligible[rollout_id] for rollout_id in risk_ids
        ),
        "n_accepted": accepted,
        "n_excluded": len(rollout_map) - accepted,
        "exclusion_reasons": dict(exclusion_reasons),
        "audit_manifest_sha256": manifest_hash,
        **_agreement_report(agreement_input),
    }
    return annotation_rows, report
