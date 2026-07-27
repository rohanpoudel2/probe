from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


Span = Tuple[int, int]


@dataclass
class TaskExample:
    """Structured example for alignment behavior tasks.

    Structured text segments let the extractor compute token-level spans for
    span-aware pooling.
    """

    example_id: str
    task_family: str
    prompt: str
    label: int
    question_id: Optional[str] = None
    condition: Optional[str] = None
    context: Optional[str] = None
    assistant_response: Optional[str] = None
    final_answer: Optional[str] = None
    chain_of_thought: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    spans: Dict[str, Span] = field(default_factory=dict)
    segments: Dict[str, str] = field(default_factory=dict)
    messages: List[Dict[str, str]] = field(default_factory=list)

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
