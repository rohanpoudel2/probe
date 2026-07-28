import numpy as np

from cli.run_task_sweep import (
    _prediction_records,
    _run_one,
    _trajectory_steps_from_view,
)


class FirstCoordinateProbe:
    scores_are_probabilities = False

    def fit(self, X, y):
        if set(y.tolist()) != {0, 1}:
            raise ValueError("both labels required")

    def score(self, X):
        return np.asarray(X)[:, 0]


def _bundle(scores, labels, prefix):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    outcome_classes = np.resize(
        np.array(
            ["target_aligned", "correct_non_target", "other_wrong_or_ambiguous"],
            dtype=object,
        ),
        len(scores),
    )
    return {
        "answer": scores[:, None],
        "labels": labels,
        "example_ids": np.asarray([f"{prefix}_e{i}" for i in range(len(labels))]),
        "question_ids": np.asarray([f"{prefix}_q{i}" for i in range(len(labels))]),
        "annotation_outcome_class": outcome_classes,
    }


def _bundle_with_outcome(scores, labels, prefix, outcomes):
    return {
        **_bundle(scores, labels, prefix),
        "annotation_outcome_class": np.asarray(outcomes, dtype=object),
    }


def test_sweep_uses_one_frozen_threshold_for_all_splits() -> None:
    train_labels = np.tile([0, 1], 4)
    train = {
        "answer": train_labels.astype(float)[:, None],
        "labels": train_labels,
        "example_ids": np.asarray([f"train_e{i}" for i in range(8)]),
        "question_ids": np.repeat([f"train_q{i}" for i in range(4)], 2),
    }
    calibration = _bundle(np.linspace(0.0, 0.99, 100), [0] * 100, "cal")
    holdout = _bundle(np.linspace(0.0, 0.98, 100), [0] * 100, "holdout")
    source_eval = _bundle([0.1, 0.2, 1.0, 1.1], [0, 0, 1, 1], "eval")
    source_test = _bundle([0.3, 0.4, 1.0, 1.1], [0, 0, 1, 1], "test")
    target_test = _bundle([0.2, 0.3, 0.995, 1.2], [0, 0, 1, 1], "target")

    row, scored, _ = _run_one(
        probe_cls=FirstCoordinateProbe,
        source_train=train,
        reference_calibration=calibration,
        reference_holdout=holdout,
        source_eval=source_eval,
        source_test=source_test,
        target_test=target_test,
        view="answer",
        k=2,
        seed=0,
        balance_mode="balanced",
        max_reference_alert_rate=0.01,
        min_reference_groups=100,
    )
    assert row["status"] == "ok"
    assert row["operating_threshold"] == np.nextafter(0.99, np.inf)
    assert row["reference_holdout_alert_rate"] == 0.0
    assert row["transfer_tpr_at_reference_alert_budget"] == 1.0
    predictions = _prediction_records(
        "run", "target_test", scored["target_test"], row["operating_threshold"]
    )
    assert {record["threshold"] for record in predictions} == {
        np.nextafter(0.99, np.inf)
    }
    assert sum(record["predicted_positive"] for record in predictions) == 2


def test_outcome_metrics_are_reported_per_slice() -> None:
    train = {
        "answer": np.array([0, 0, 1, 1])[:, None],
        "labels": np.array([0, 1, 0, 1]),
        "example_ids": np.array(["train_e0", "train_e1", "train_e2", "train_e3"]),
        "question_ids": np.array(["train_q0", "train_q0", "train_q1", "train_q1"]),
        "annotation_outcome_class": np.array(
            [
                "target_aligned",
                "target_aligned",
                "correct_non_target",
                "other_wrong_or_ambiguous",
            ]
        ),
    }
    reference = _bundle(
        np.linspace(0.0, 0.99, 100), [0] * 100, "cal"
    )
    holdout = _bundle(np.linspace(0.0, 0.98, 100), [0] * 100, "holdout")
    source_eval = _bundle(
        [0.1, 0.5, 0.2, 0.9],
        [0, 1, 1, 1],
        "eval",
    )
    target_test = _bundle_with_outcome(
        [1.2, 0.2, 1.1, 1.3],
        [1, 0, 1, 1],
        "target",
        [
            "target_aligned",
            "correct_non_target",
            "other_wrong_or_ambiguous",
            "other_wrong_or_ambiguous",
        ],
    )
    row, _, _ = _run_one(
        probe_cls=FirstCoordinateProbe,
        source_train=train,
        reference_calibration=reference,
        reference_holdout=holdout,
        source_eval=source_eval,
        source_test=None,
        target_test=target_test,
        view="answer",
        k=2,
        seed=0,
        balance_mode="balanced",
        max_reference_alert_rate=0.01,
        min_reference_groups=100,
    )
    assert row["status"] == "ok"
    assert (
        row["transfer_target_aligned_tpr_at_reference_alert_budget"] == 1.0
    )
    assert (
        row["transfer_other_wrong_or_ambiguous_tpr_at_reference_alert_budget"]
        == 1.0
    )
    assert np.isnan(row["transfer_correct_non_target_tpr_at_reference_alert_budget"])


def test_selection_only_does_not_score_behavior_test_splits() -> None:
    labels = np.tile([0, 1], 4)
    train = {
        "answer": labels.astype(float)[:, None],
        "labels": labels,
        "example_ids": np.asarray([f"train_e{i}" for i in range(8)]),
        "question_ids": np.repeat([f"train_q{i}" for i in range(4)], 2),
    }
    reference = _bundle(np.linspace(0.0, 0.99, 100), [0] * 100, "cal")
    holdout = _bundle(np.linspace(0.0, 0.98, 100), [0] * 100, "holdout")
    source_eval = _bundle([0.1, 0.2, 1.0, 1.1], [0, 0, 1, 1], "eval")
    row, scored, _ = _run_one(
        probe_cls=FirstCoordinateProbe,
        source_train=train,
        reference_calibration=reference,
        reference_holdout=holdout,
        source_eval=source_eval,
        source_test=None,
        target_test=None,
        view="answer",
        k=2,
        seed=0,
        balance_mode="balanced",
        max_reference_alert_rate=0.01,
        min_reference_groups=100,
        selection_only=True,
    )
    assert row["status"] == "ok"
    assert np.isnan(row["test_tpr_at_reference_alert_budget"])
    assert set(scored) == {
        "reference_calibration",
        "reference_holdout",
        "source_eval",
    }


def test_trajectory_steps_from_view_prefers_bundle_metadata() -> None:
    bundle = {
        "trajectory_prefix_percentiles": np.array("[10, 50, 90]", dtype=str),
        "trajectory_prefix_stack_p10": np.zeros((1, 1)),
        "trajectory_prefix_stack_p50": np.zeros((1, 1)),
        "trajectory_prefix_stack_p90": np.zeros((1, 1)),
    }
    assert _trajectory_steps_from_view("trajectory_prefix_stack_p10", bundle) == 1
    assert _trajectory_steps_from_view("trajectory_prefix_stack_p50", bundle) == 2


def test_trajectory_steps_from_view_fallbacks_to_prefix_view_names() -> None:
    bundle = {
        "trajectory_prefix_p10": np.zeros((1, 1)),
        "trajectory_prefix_p50": np.zeros((1, 1)),
        "trajectory_prefix_p100": np.zeros((1, 1)),
        "trajectory_prefix_stack_p50": np.zeros((1, 1)),
    }
    assert _trajectory_steps_from_view("trajectory_prefix_stack_p50", bundle) == 2
