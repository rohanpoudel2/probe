from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from data.schema import TaskExample


@dataclass
class TaskSpec:
    name: str
    primary_metric: str
    label_semantics: Dict[int, str]
    grouped_split_key: str = "question_id"
    default_spans: List[str] = field(default_factory=lambda: ["full_text", "answer"])
    notes: str = ""


class BehaviorTask(ABC):
    spec: TaskSpec

    @abstractmethod
    def load(self, path: Optional[str] = None) -> List[TaskExample]:
        """Return structured examples for this behavior family."""

    def audit(self, examples: Iterable[TaskExample]) -> Dict[str, object]:
        examples = list(examples)
        pos = sum(int(ex.label == 1) for ex in examples)
        neg = len(examples) - pos
        key = self.spec.grouped_split_key
        grouped_values = [getattr(ex, key, None) or ex.metadata.get(key) for ex in examples]
        non_null_groups = {g for g in grouped_values if g is not None}
        return {
            "task": self.spec.name,
            "n_examples": len(examples),
            "n_positive": pos,
            "n_negative": neg,
            "positive_rate": (pos / len(examples)) if examples else 0.0,
            "grouped_split_key": key,
            "n_groups": len(non_null_groups),
            "default_spans": list(self.spec.default_spans),
            "notes": self.spec.notes,
        }
