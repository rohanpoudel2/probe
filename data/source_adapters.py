from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _list_to_text(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    parts: list[str] = []
    for item in messages:
        if isinstance(item, dict):
            content = item.get("content") or item.get("text") or item.get("message")
            role = item.get("role") or item.get("type")
            content_text = ensure_text(content)
            if not content_text:
                continue
            parts.append(f"{role}: {content_text}" if role else content_text)
        else:
            item_text = ensure_text(item)
            if item_text:
                parts.append(item_text)
    return "\n".join(parts) if parts else None


def _flatten_prompt(value: Any) -> str | None:
    if isinstance(value, str):
        return ensure_text(value)
    if isinstance(value, list):
        return _list_to_text(value)
    if isinstance(value, dict):
        for key in ("content", "text", "message", "prompt", "question"):
            text = ensure_text(value.get(key))
            if text:
                return text
    return None


def first_text(row: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        text = _flatten_prompt(row.get(key))
        if text:
            return text
    return None


def normalize_answer_text(value: Any) -> str:
    if isinstance(value, list):
        parts = [ensure_text(item) for item in value]
        parts = [part for part in parts if part]
        return parts[0] if parts else "unknown"
    return ensure_text(value) or "unknown"


def pick_wrong_answer(correct: str, choices: list[str] | None = None) -> str:
    if choices:
        for choice in choices:
            if choice and choice != correct:
                return choice
    swaps = {"true": "false", "false": "true", "yes": "no", "no": "yes"}
    return swaps.get(correct.lower(), f"Not {correct}")


def choice_list(row: dict[str, Any]) -> list[str]:
    value = (
        row.get("choices")
        or row.get("options")
        or row.get("candidates")
        or row.get("answer_choices")
    )
    if isinstance(value, list):
        choices: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                choices.append(item.strip())
            elif isinstance(item, dict):
                text = first_text(item, ("text", "content", "label", "option"))
                if text:
                    choices.append(text)
        return choices
    if isinstance(value, dict):
        for key in ("text", "choices", "options"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [
                    item.strip()
                    for item in nested
                    if isinstance(item, str) and item.strip()
                ]
        return [
            item.strip()
            for key in sorted(value)
            if isinstance((item := value[key]), str) and item.strip()
        ]
    return []


def resolve_correct_choice(row: dict[str, Any]) -> tuple[str, list[str]]:
    choices = choice_list(row)
    answer = next(
        (
            row[key]
            for key in ("answer", "answerKey", "correct")
            if row.get(key) is not None
        ),
        None,
    )
    raw_choices = row.get("choices")
    if isinstance(raw_choices, dict) and answer is not None:
        labels = raw_choices.get("label")
        texts = raw_choices.get("text")
        if (
            isinstance(labels, list)
            and isinstance(texts, list)
            and len(labels) == len(texts)
        ):
            label_strings = [str(label).strip() for label in labels]
            answer_key = str(answer).strip()
            if answer_key in label_strings:
                resolved = ensure_text(texts[label_strings.index(answer_key)])
                if resolved:
                    return resolved, choices
    if isinstance(answer, int) and 0 <= answer < len(choices):
        return choices[answer], choices
    if isinstance(answer, str):
        answer_text = answer.strip()
        if answer_text.isdigit():
            index = int(answer_text)
            if 0 <= index < len(choices):
                return choices[index], choices
        if len(answer_text) == 1 and answer_text.isalpha() and choices:
            index = ord(answer_text.upper()) - ord("A")
            if 0 <= index < len(choices):
                return choices[index], choices
        if answer_text:
            return answer_text, choices
    for key in ("correct_answer", "gold", "gold_answer", "ground_truth", "solution"):
        text = ensure_text(row.get(key))
        if text:
            return text, choices
    return "unknown", choices


def format_mcq_prompt(question: str, choices: list[str]) -> str:
    if not choices:
        return question
    rendered = [
        f"{chr(ord('A') + index)}. {choice}"
        for index, choice in enumerate(choices)
    ]
    return f"{question}\n" + "\n".join(rendered)
