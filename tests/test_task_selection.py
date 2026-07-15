import pandas as pd

from evaluation.task_selection import apply_frozen_selection, select_frozen_source_systems


def test_layer_view_is_selected_at_fixed_k_and_reused_across_k_and_targets() -> None:
    rows = []
    for target in ("t1", "t2"):
        for k in (1, 8):
            rows.extend(
                [
                    {
                        "model": "m",
                        "source_task": "s",
                        "target_task": target,
                        "probe": "p",
                        "balance_mode": "balanced",
                        "layer": 1,
                        "view": "answer",
                        "k": k,
                        "eval_recall_at_1pct_fpr_mean": 0.9 if k == 1 else 0.2,
                    },
                    {
                        "model": "m",
                        "source_task": "s",
                        "target_task": target,
                        "probe": "p",
                        "balance_mode": "balanced",
                        "layer": 2,
                        "view": "answer",
                        "k": k,
                        "eval_recall_at_1pct_fpr_mean": 0.1 if k == 1 else 0.8,
                    },
                ]
            )
    summary = pd.DataFrame(rows)
    selected = select_frozen_source_systems(summary, selection_k=8)
    assert selected["layer"].tolist() == [2]
    report = apply_frozen_selection(summary, selected)
    assert set(report["k"]) == {1, 8}
    assert set(report["target_task"]) == {"t1", "t2"}
    assert set(report["layer"]) == {2}
