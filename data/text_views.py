from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from data.schema import TaskExample


ALLOWED_TEXT_VIEWS = {"prompt_text", "answer_text", "transcript_text"}
COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")


def text_view(example: TaskExample, view: str) -> str:
    if view not in ALLOWED_TEXT_VIEWS:
        raise ValueError(f"Unknown text view: {view}")
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
    answer = example.assistant_response or example.final_answer or ""
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
