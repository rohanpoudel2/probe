from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


SCENARIO_SCHEMA_VERSION = "frontier-monitor-scenario-v2"
ROLLOUT_SCHEMA_VERSION = "frontier-monitor-rollout-v2"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _scenario_hash_payload(
    *,
    task_family: str,
    group_id: str,
    messages: list[dict[str, str]],
    condition: str,
    protocol_split: str,
    source: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_family": task_family,
        "group_id": group_id,
        "messages": messages,
        "condition": condition,
        "protocol_split": protocol_split,
        "source": source,
        "metadata": metadata,
    }


def validate_messages(messages: Any, *, allow_assistant: bool) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        allowed_roles = (
            {"system", "user", "assistant"} if allow_assistant else {"system", "user"}
        )
        if role not in allowed_roles:
            raise ValueError(
                f"messages[{index}].role must be one of {sorted(allowed_roles)}"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content must be non-empty text")
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != ("assistant" if allow_assistant else "user"):
        expected = "assistant" if allow_assistant else "user"
        raise ValueError(f"The final message must have role={expected!r}")
    return normalized


@dataclass(frozen=True)
class ScenarioRecord:
    scenario_id: str
    group_id: str
    task_family: str
    messages: list[dict[str, str]]
    condition: str
    protocol_split: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCENARIO_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCENARIO_SCHEMA_VERSION:
            raise ValueError(f"Unsupported scenario schema: {self.schema_version}")
        if not all(
            str(value).strip()
            for value in (
                self.scenario_id,
                self.group_id,
                self.task_family,
                self.condition,
                self.source,
            )
        ):
            raise ValueError(
                "Scenario identifiers, condition, and source must be non-empty"
            )
        if self.protocol_split not in {"train", "calibration", "eval", "test"}:
            raise ValueError("protocol_split must be train, calibration, eval, or test")
        if not isinstance(self.metadata, dict):
            raise ValueError("scenario metadata must be an object")
        object.__setattr__(
            self,
            "messages",
            validate_messages(self.messages, allow_assistant=False),
        )

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ScenarioRecord":
        prohibited = {
            "label",
            "assistant_response",
            "response_text",
            "final_answer",
            "reasoning",
            "chain_of_thought",
        }
        present = sorted(key for key in prohibited if row.get(key) is not None)
        if present:
            raise ValueError(
                f"Scenario {row.get('scenario_id', '<unknown>')} contains authored outcome fields {present}"
            )
        schema_version = row.get("schema_version", SCENARIO_SCHEMA_VERSION)
        if schema_version != SCENARIO_SCHEMA_VERSION:
            raise ValueError(f"Unsupported scenario schema: {schema_version}")
        scenario_id = str(row.get("scenario_id", "")).strip()
        group_id = str(row.get("group_id", "")).strip()
        task_family = str(row.get("task_family", "")).strip()
        condition = str(row.get("condition", "")).strip()
        protocol_split = str(row.get("protocol_split", "")).strip()
        source = str(row.get("source", "")).strip()
        if not all(
            (scenario_id, group_id, task_family, condition, protocol_split, source)
        ):
            raise ValueError(
                "scenario_id, group_id, task_family, condition, protocol_split, and source are required"
            )
        if protocol_split not in {"train", "calibration", "eval", "test"}:
            raise ValueError("protocol_split must be train, calibration, eval, or test")
        messages = validate_messages(row.get("messages"), allow_assistant=False)
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("scenario metadata must be an object")
        record = cls(
            scenario_id=scenario_id,
            group_id=group_id,
            task_family=task_family,
            messages=messages,
            condition=condition,
            protocol_split=protocol_split,
            source=source,
            metadata=metadata,
            schema_version=schema_version,
        )
        supplied_hash = row.get("scenario_hash")
        if supplied_hash is not None and supplied_hash != record.scenario_hash():
            raise ValueError(
                f"Scenario {scenario_id} has a stale or invalid scenario_hash"
            )
        return record

    def scenario_hash(self) -> str:
        return content_hash(
            _scenario_hash_payload(
                task_family=self.task_family,
                group_id=self.group_id,
                messages=self.messages,
                condition=self.condition,
                protocol_split=self.protocol_split,
                source=self.source,
                metadata=self.metadata,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "group_id": self.group_id,
            "task_family": self.task_family,
            "messages": self.messages,
            "condition": self.condition,
            "protocol_split": self.protocol_split,
            "source": self.source,
            "metadata": self.metadata,
            "scenario_hash": self.scenario_hash(),
        }


@dataclass(frozen=True)
class RolloutRecord:
    rollout_id: str
    scenario: ScenarioRecord
    response_text: str
    messages: list[dict[str, str]]
    model_id: str
    model_revision: str
    tokenizer_revision: str
    seed: int
    generation: dict[str, Any]
    provenance: dict[str, Any]
    reasoning: str | None = None
    final_answer: str | None = None
    schema_version: str = ROLLOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.rollout_id or not self.response_text.strip():
            raise ValueError("rollout_id and non-empty response_text are required")
        if not self.model_revision or self.model_revision in {
            "main",
            "latest",
            "unpinned",
        }:
            raise ValueError(
                "model_revision must be an immutable commit or tag, not main/latest/unpinned"
            )
        if not self.tokenizer_revision or self.tokenizer_revision in {
            "main",
            "latest",
            "unpinned",
        }:
            raise ValueError(
                "tokenizer_revision must be immutable, not main/latest/unpinned"
            )
        validate_messages(self.messages, allow_assistant=True)
        if self.messages[:-1] != self.scenario.messages:
            raise ValueError("Rollout prompt messages differ from the source scenario")
        if self.messages[-1]["content"] != self.response_text:
            raise ValueError("Final assistant message must equal response_text")
        if "confidence_trace" in self.generation:
            from data.generation_confidence import validate_generation_confidence_trace

            token_ids = self.generation.get("response_token_ids")
            token_count = self.generation.get("response_token_count")
            if not isinstance(token_ids, list) or token_count != len(token_ids):
                raise ValueError(
                    "Generation confidence requires aligned response token IDs/count"
                )
            validate_generation_confidence_trace(
                self.generation["confidence_trace"],
                expected_token_count=token_count,
                expected_token_ids=token_ids,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rollout_id": self.rollout_id,
            "scenario_id": self.scenario.scenario_id,
            "group_id": self.scenario.group_id,
            "task_family": self.scenario.task_family,
            "condition": self.scenario.condition,
            "protocol_split": self.scenario.protocol_split,
            "source": self.scenario.source,
            "scenario_hash": self.scenario.to_dict()["scenario_hash"],
            "prompt_messages": self.scenario.messages,
            "messages": self.messages,
            "response_text": self.response_text,
            "reasoning": self.reasoning,
            "final_answer": self.final_answer,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "seed": self.seed,
            "generation": self.generation,
            "provenance": self.provenance,
            "metadata": self.scenario.metadata,
            "data_origin": "on_policy_generation",
            "generated_by_model": True,
        }
