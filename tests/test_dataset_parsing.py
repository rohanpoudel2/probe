from scripts.build_exact_paper_datasets import (
    _choice_list,
    _resolve_correct_choice,
)


def test_arc_choice_dictionary_is_resolved() -> None:
    row = {
        "choices": {"label": ["A", "B", "C"], "text": ["alpha", "beta", "gamma"]},
        "answerKey": "B",
    }
    assert _choice_list(row) == ["alpha", "beta", "gamma"]
    assert _resolve_correct_choice(row) == ("beta", ["alpha", "beta", "gamma"])


def test_aqua_correct_field_is_resolved() -> None:
    row = {"options": ["A) one", "B) two"], "correct": "B"}
    assert _resolve_correct_choice(row)[0] == "B) two"
