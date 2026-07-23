from cli.generate_task_rollouts import (
    _generation_stop_reason,
    _split_reasoning,
)


def test_reasoning_is_split_before_special_tokens_are_removed() -> None:
    reasoning, answer = _split_reasoning(
        "<s><think>Check both choices carefully.</think>The answer is B.</s>",
        cleaned_response="Check both choices carefully.The answer is B.",
        special_tokens=("<s>", "</s>", "<think>", "</think>"),
    )
    assert reasoning == "Check both choices carefully."
    assert answer == "The answer is B."


def test_bracketed_reasoning_tags_are_supported() -> None:
    reasoning, answer = _split_reasoning(
        "[THINK]Work through the evidence.[/THINK]Final response.",
        cleaned_response="Work through the evidence.Final response.",
        special_tokens=("[THINK]", "[/THINK]"),
    )
    assert reasoning == "Work through the evidence."
    assert answer == "Final response."


def test_generation_stop_reason_distinguishes_eos_from_length_cap() -> None:
    assert (
        _generation_stop_reason(
            [10, 11, 2], eos_token_ids=2, max_new_tokens=3
        )
        == "eos_token"
    )
    assert (
        _generation_stop_reason(
            [10, 11, 12], eos_token_ids=2, max_new_tokens=3
        )
        == "max_new_tokens"
    )
