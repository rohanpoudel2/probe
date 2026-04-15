from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


Span = Tuple[int, int]


@dataclass
class TaskExample:
    """Structured example for alignment behavior tasks.

    Phase 1 extends the Phase 0 schema with explicit named text segments.
    The extractor uses these segments to compute token-level spans for
    span-aware pooling.
    """

    example_id: str
    task_family: str
    prompt: str
    label: int
    question_id: Optional[str] = None
    turn_id: Optional[int] = None
    condition: Optional[str] = None
    context: Optional[str] = None
    assistant_response: Optional[str] = None
    final_answer: Optional[str] = None
    chain_of_thought: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    spans: Dict[str, Span] = field(default_factory=dict)
    segments: Dict[str, str] = field(default_factory=dict)

    def build_segments(self) -> Dict[str, str]:
        if self.segments:
            return self.segments

        segments: Dict[str, str] = {}
        if self.context:
            segments["context"] = self.context
        segments["prompt"] = self.prompt
        if self.chain_of_thought:
            segments["reasoning"] = self.chain_of_thought
        if self.final_answer:
            segments["answer"] = self.final_answer
        elif self.assistant_response:
            segments["answer"] = self.assistant_response
        self.segments = segments
        return segments

    def full_text(self) -> str:
        ordered = [text for _, text in self.build_segments().items() if text]
        return "\n\n".join(ordered)

    def to_record(self) -> Dict[str, Any]:
        return {
            "text": self.full_text(),
            "label": self.label,
            "example_id": self.example_id,
            "question_id": self.question_id,
            "condition": self.condition,
            "task_family": self.task_family,
            **self.metadata,
        }
