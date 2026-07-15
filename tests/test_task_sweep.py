import numpy as np

from cli.run_task_sweep import _prediction_records, _run_one


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
    return {
        "answer": scores[:, None],
        "labels": labels,
        "example_ids": np.asarray([f"{prefix}_e{i}" for i in range(len(labels))]),
        "question_ids": np.asarray([f"{prefix}_q{i}" for i in range(len(labels))]),
    }


def test_sweep_uses_one_frozen_threshold_for_all_splits() -> None:
    train_labels = np.tile([0, 1], 4)
    train = {
        "answer": train_labels.astype(float)[:, None],
        "labels": train_labels,
        "example_ids": np.asarray([f"train_e{i}" for i in range(8)]),
        "question_ids": np.repeat([f"train_q{i}" for i in range(4)], 2),
    }
    calibration = _bundle(
        np.concatenate([np.linspace(0.0, 0.99, 100), [1.2, 1.3]]),
        [0] * 100 + [1, 1],
        "cal",
    )
    source_eval = _bundle([0.1, 0.2, 1.0, 1.1], [0, 0, 1, 1], "eval")
    source_test = _bundle([0.3, 0.4, 1.0, 1.1], [0, 0, 1, 1], "test")
    target_test = _bundle([0.2, 0.3, 0.995, 1.2], [0, 0, 1, 1], "target")

    row, scored, _ = _run_one(
        probe_cls=FirstCoordinateProbe,
        source_train=train,
        source_calibration=calibration,
        source_eval=source_eval,
        source_test=source_test,
        target_test=target_test,
        view="answer",
        k=2,
        seed=0,
        balance_mode="balanced",
        max_fpr=0.01,
        min_calibration_negatives=100,
    )
    assert row["status"] == "ok"
    assert row["operating_threshold"] == 0.99
    assert row["transfer_recall_at_frozen_fpr"] == 1.0
    predictions = _prediction_records(
        "run", "target_test", scored["target_test"], row["operating_threshold"]
    )
    assert {record["threshold"] for record in predictions} == {0.99}
    assert sum(record["predicted_positive"] for record in predictions) == 2
