from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from data.schema import TaskExample


@dataclass
class TaskSpec:
    name: str
    label_semantics: Dict[int, str]
    grouped_split_key: str = "question_id"
    evaluation_only: bool = False
    unavailable_baselines: Dict[str, str] = field(default_factory=dict)


class BehaviorTask(ABC):
    spec: TaskSpec

    @abstractmethod
    def load(self, path: Optional[str] = None) -> List[TaskExample]:
        """Return structured examples for this behavior family."""
