from __future__ import annotations

import re
from math import ceil
from typing import Iterable

import numpy as np

from data.schema import TaskExample


ALLOWED_TEXT_VIEWS = {"prompt_text", "answer_text", "transcript_text"}
RESPONSE_PREFIX_TEXT_VIEW_RE = re.compile(r"^response_prefix_text_p(\d+)$")
COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")


def response_prefix_text_view(percentile: int) -> str:
    percentile = int(percentile)
    if not 1 <= percentile <= 100:
        raise ValueError("response-prefix percentiles must be in [1, 100]")
    return f"response_prefix_text_p{percentile}"


def parse_response_prefix_text_view(view: str) -> int | None:
    match = RESPONSE_PREFIX_TEXT_VIEW_RE.fullmatch(str(view))
    if match is None:
        return None
    percentile = int(match.group(1))
    return percentile if 1 <= percentile <= 100 else None


def is_valid_text_view(view: str) -> bool:
    return view in ALLOWED_TEXT_VIEWS or parse_response_prefix_text_view(view) is not None


def _visible_answer(example: TaskExample) -> str:
    if example.messages:
        return "\n".join(
            message["content"]
            for message in example.messages
            if message.get("role") == "assistant"
        )
    return example.assistant_response or example.final_answer or ""


def text_view(example: TaskExample, view: str) -> str:
    if not is_valid_text_view(view):
        raise ValueError(f"Unknown text view: {view}")
    prefix_percentile = parse_response_prefix_text_view(view)
    if prefix_percentile is not None:
        answer = _visible_answer(example)
        if not answer:
            return ""
        end = max(1, min(len(answer), ceil(len(answer) * prefix_percentile / 100)))
        return answer[:end]
    if example.messages:
        if view == "prompt_text":
            return "\n".join(
                message["content"]
                for message in example.messages
                if message.get("role") != "assistant"
            )
        if view == "answer_text":
            return "\n".join(
                message["content"]
                for message in example.messages
                if message.get("role") == "assistant"
            )
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in example.messages
        )

    prompt = "\n\n".join(part for part in (example.context, example.prompt) if part)
    answer = _visible_answer(example)
    reasoning = example.chain_of_thought or ""
    if view == "prompt_text":
        return prompt
    if view == "answer_text":
        return answer
    return "\n\n".join(part for part in (prompt, reasoning, answer) if part)


def examples_to_text_arrays(
    examples: Iterable[TaskExample], view: str
) -> dict[str, np.ndarray]:
    rows = list(examples)
    texts = np.asarray([text_view(example, view) for example in rows], dtype=str)
    if not len(rows):
        raise ValueError("Cannot build a text view from an empty dataset")
    if np.any(np.char.str_len(texts) == 0):
        empty = [rows[index].example_id for index in np.flatnonzero(np.char.str_len(texts) == 0)]
        raise ValueError(f"View {view} contains empty text for examples {empty[:5]}")
    return {
        "texts": texts,
        "labels": np.asarray([example.label for example in rows], dtype=np.int64),
        "example_ids": np.asarray([example.example_id for example in rows], dtype=str),
        "question_ids": np.asarray(
            [example.question_id or example.example_id for example in rows], dtype=str
        ),
        "protocol_splits": np.asarray(
            [str(example.metadata.get("protocol_split") or "") for example in rows],
            dtype=str,
        ),
    }


def monitored_model_identity(examples: Iterable[TaskExample]) -> dict[str, str]:
    rows = list(examples)
    identities = {
        (
            str(example.metadata.get("model_id") or ""),
            str(example.metadata.get("model_revision") or ""),
            str(example.metadata.get("tokenizer_revision") or ""),
        )
        for example in rows
    }
    if len(identities) != 1:
        raise ValueError(
            "A text cache must contain outputs from exactly one monitored model/revision"
        )
    model_id, model_revision, tokenizer_revision = identities.pop()
    if not model_id:
        raise ValueError("On-policy rows lack model_id provenance")
    for name, revision in (
        ("model_revision", model_revision),
        ("tokenizer_revision", tokenizer_revision),
    ):
        if not COMMIT_RE.fullmatch(revision):
            raise ValueError(f"On-policy rows lack an immutable {name}")
    return {
        "monitored_model_id": model_id,
        "monitored_model_revision": model_revision,
        "monitored_tokenizer_revision": tokenizer_revision,
    }
