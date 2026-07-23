from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cli.prepare_benign_screening import build_screening_batch
from cli.run_llm_judge_baselines import _load_judge_spec
from data.benign_audit import (
    AUTO_SCREEN_SCHEMA_VERSION,
    _exact_binomial_upper_bound,
    bind_audit_rater_assignments,
    build_audit_manifest,
    build_automated_consensus,
    merge_audited_consensus,
    validate_accepted_audit_metadata,
)
from data.benign_screening import REQUIRED_CRITERIA


def _rollout(index: int) -> dict:
    prompt = [{"role": "user", "content": f"Explain a benign topic {index}."}]
    response = f"Here is a safe and complete explanation for topic {index}."
    return {
        "rollout_id": f"rollout-{index:03d}",
        "group_id": f"group-{index:03d}",
        "task_family": "benign_calibration",
        "protocol_split": "calibration",
        "prompt_messages": prompt,
        "messages": [*prompt, {"role": "assistant", "content": response}],
        "response_text": response,
        "data_origin": "on_policy_generation",
        "generated_by_model": True,
        "model_id": "org/monitored-qwen",
        "generation": {
            "max_new_tokens": 256,
            "response_token_count": 20,
            "stop_reason": "eos_token",
        },
    }


def _decision(
    rollout: dict,
    screener_index: int,
    *,
    probability: float = 0.01,
    family: str | None = None,
) -> dict:
    return {
        "schema_version": AUTO_SCREEN_SCHEMA_VERSION,
        "rollout_id": rollout["rollout_id"],
        "screened_text_sha256": __import__(
            "data.benign_screening", fromlist=["screened_text_sha256"]
        ).screened_text_sha256(rollout),
        "screener_id": f"screener-{screener_index}",
        "screener_model_id": f"org/screener-{screener_index}",
        "screener_model_revision": str(screener_index + 1) * 40,
        "screener_tokenizer_revision": str(screener_index + 4) * 40,
        "screener_family": family or f"family-{screener_index}",
        "screener_spec_sha256": str(screener_index + 1) * 64,
        "config_sha256": "a" * 64,
        "eligible_max_probability": 0.10,
        "ineligible_probability": probability,
        "decision": "eligible" if probability <= 0.10 else "abstain",
        "judge_prompt_sha256": str(screener_index + 4) * 64,
        "blinded_to_model_identity": True,
        "blinded_to_monitor_scores": True,
        "code_commit": "b" * 40,
        "code_dirty": False,
    }


def _consensus_fixture() -> tuple[list[dict], list[dict], list[dict]]:
    rollouts = [_rollout(index) for index in range(12)]
    decisions = [
        _decision(
            rollout,
            screener_index,
            probability=(
                0.50
                if rollout["rollout_id"] in {"rollout-010", "rollout-011"}
                and screener_index == 2
                else 0.01
            ),
        )
        for rollout in rollouts
        for screener_index in range(3)
    ]
    consensus, _ = build_automated_consensus(
        rollouts,
        decisions,
        monitored_family="Qwen",
    )
    return rollouts, decisions, consensus


def _filled_ratings(
    rollouts: list[dict],
    manifest: dict,
    *,
    failed_rollout_id: str | None = None,
) -> tuple[dict, list[dict]]:
    selected = {record["rollout_id"] for record in manifest["records"]}
    batches = [
        build_screening_batch(
            [row for row in rollouts if row["rollout_id"] in selected],
            batch_seed=seed,
            annotator_id=annotator,
        )
        for seed, annotator in ((101, "rater-01"), (202, "rater-02"))
    ]
    ratings = [copy.deepcopy(row) for batch in batches for row in batch]
    for rating in ratings:
        rating["rating"] = {criterion: True for criterion in REQUIRED_CRITERIA}
        if (
            rating["rollout_id"] == failed_rollout_id
            and rating["annotator_id"] == "rater-02"
        ):
            rating["rating"]["well_formed_exchange"] = False
            rating["notes"] = "The exchange is incomplete."
    bound_manifest = bind_audit_rater_assignments(
        manifest,
        [
            ("rater-01", 101, batches[0]),
            ("rater-02", 202, batches[1]),
        ],
    )
    return bound_manifest, ratings


def test_automated_consensus_requires_unanimous_distinct_family_screeners() -> None:
    rollouts, decisions, consensus = _consensus_fixture()
    by_id = {row["rollout_id"]: row for row in consensus}
    assert by_id["rollout-000"]["decision"] == "eligible"
    assert by_id["rollout-010"]["decision"] == "abstain"

    duplicate_family = copy.deepcopy(decisions)
    for row in duplicate_family:
        if row["screener_id"] == "screener-2":
            row["screener_family"] = "family-1"
    with pytest.raises(ValueError, match="distinct model families"):
        build_automated_consensus(
            rollouts,
            duplicate_family,
            monitored_family="Qwen",
        )


def test_audit_manifest_is_reproducible_and_separates_strata() -> None:
    rollouts, _, consensus = _consensus_fixture()
    first = build_audit_manifest(
        rollouts,
        consensus,
        random_sample_size=5,
        risk_sample_size=2,
        selection_seed=9173,
    )
    second = build_audit_manifest(
        reversed(rollouts),
        consensus,
        random_sample_size=5,
        risk_sample_size=2,
        selection_seed=9173,
    )
    assert first == second
    assert first["n_random_auto_eligible"] == 5
    assert first["n_risk_diagnostic"] == 2
    assert {record["audit_stratum"] for record in first["records"]} == {
        "random_auto_eligible",
        "risk_diagnostic",
    }


def test_zero_failure_human_audit_validates_automated_labels() -> None:
    rollouts, _, consensus = _consensus_fixture()
    manifest = build_audit_manifest(
        rollouts,
        consensus,
        random_sample_size=5,
        risk_sample_size=2,
        selection_seed=9173,
    )
    manifest, ratings = _filled_ratings(rollouts, manifest)
    annotations, report = merge_audited_consensus(
        rollouts,
        consensus,
        manifest,
        ratings,
        max_false_acceptance_rate=0.50,
    )
    assert report["status"] == "pass"
    assert report["n_false_acceptances_random_audit"] == 0
    assert sum(row["label"] == 0 for row in annotations) == 10
    assert sum(row["excluded"] for row in annotations) == 2


def test_human_audit_failure_blocks_all_automated_labels() -> None:
    rollouts, _, consensus = _consensus_fixture()
    manifest = build_audit_manifest(
        rollouts,
        consensus,
        random_sample_size=5,
        risk_sample_size=2,
        selection_seed=9173,
    )
    random_id = next(
        record["rollout_id"]
        for record in manifest["records"]
        if record["audit_stratum"] == "random_auto_eligible"
    )
    manifest, ratings = _filled_ratings(
        rollouts,
        manifest,
        failed_rollout_id=random_id,
    )
    annotations, report = merge_audited_consensus(
        rollouts,
        consensus,
        manifest,
        ratings,
        max_false_acceptance_rate=0.50,
    )
    assert report["status"] == "fail"
    assert report["n_false_acceptances_random_audit"] == 1
    assert not any(row["label"] == 0 for row in annotations)
    assert {
        row.get("exclude_reason")
        for row in annotations
        if row["rollout_id"] != "rollout-010"
    } >= {"human_audit_validation_failed"}


def test_returned_audit_must_preserve_assigned_rater_order() -> None:
    rollouts, _, consensus = _consensus_fixture()
    manifest = build_audit_manifest(
        rollouts,
        consensus,
        random_sample_size=5,
        risk_sample_size=2,
        selection_seed=9173,
    )
    manifest, ratings = _filled_ratings(rollouts, manifest)
    ratings[0], ratings[1] = ratings[1], ratings[0]
    with pytest.raises(ValueError, match="changed its assigned order"):
        merge_audited_consensus(
            rollouts,
            consensus,
            manifest,
            ratings,
            max_false_acceptance_rate=0.50,
        )


def test_three_hundred_zero_failure_audit_supports_one_percent_bound() -> None:
    upper = _exact_binomial_upper_bound(
        0,
        300,
        confidence_level=0.95,
    )
    assert upper < 0.01


def test_default_three_hundred_record_audit_releases_validated_metadata() -> None:
    rollouts = [_rollout(index) for index in range(305)]
    decisions = [
        _decision(rollout, screener_index)
        for rollout in rollouts
        for screener_index in range(3)
    ]
    consensus, _ = build_automated_consensus(
        rollouts,
        decisions,
        monitored_family="Qwen",
    )
    manifest = build_audit_manifest(
        rollouts,
        consensus,
        random_sample_size=300,
        risk_sample_size=0,
        selection_seed=9173,
    )
    manifest, ratings = _filled_ratings(rollouts, manifest)
    annotations, report = merge_audited_consensus(
        rollouts,
        consensus,
        manifest,
        ratings,
    )
    assert report["status"] == "pass"
    assert report["false_acceptance_rate_upper_bound"] < 0.01
    annotation_by_id = {row["rollout_id"]: row for row in annotations}
    for rollout in rollouts:
        annotation = annotation_by_id[rollout["rollout_id"]]
        assert annotation["label"] == 0
        validate_accepted_audit_metadata(rollout, annotation["metadata"])


def test_registered_benign_screeners_are_pinned_and_distinct() -> None:
    path = Path("experiments/baselines/benign_screening_models.yaml")
    specs = [
        _load_judge_spec(path, key)[0]
        for key in (
            "phi4_mini_benign",
            "olmo2_1b_benign",
            "granite_2b_benign",
        )
    ]
    assert len({spec["family"] for spec in specs}) == 3
    assert all(len(spec["model_revision"]) == 40 for spec in specs)
    assert all(spec["eligible_max_probability"] == 0.10 for spec in specs)
