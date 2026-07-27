from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.schema import TaskExample
from extraction.task_extractor import TaskActivationExtractor, TaskExtractionConfig


class CharacterTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    chat_template = "fixture"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class HiddenStateFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(num_hidden_layers=2)

    def forward(self, input_ids, attention_mask, output_hidden_states):
        length = input_ids.shape[1]
        shape = (1, length, 3)
        return SimpleNamespace(
            hidden_states=(
                torch.zeros(shape, device=input_ids.device),
                torch.ones(shape, device=input_ids.device),
                torch.full(shape, 2.0, device=input_ids.device),
            )
        )


def _extractor(**overrides):
    values = dict(
        model_name="fixture",
        layers=[0, 1],
        views=["answer"],
        use_chat_template=True,
        require_model_generated=False,
    )
    values.update(overrides)
    cfg = TaskExtractionConfig(**values)
    extractor = object.__new__(TaskActivationExtractor)
    extractor.cfg = cfg
    extractor.tokenizer = CharacterTokenizer()
    extractor.model = HiddenStateFixture()
    return extractor


def _example() -> TaskExample:
    return TaskExample(
        example_id="r1",
        task_family="test",
        prompt="Question",
        final_answer="Answer",
        label=1,
        question_id="q1",
        segments={"prompt": "Question", "answer": "Answer"},
        messages=[
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
    )


def test_chat_spans_locate_exact_assistant_answer() -> None:
    extractor = _extractor()
    encoded = extractor._tokenize_segments(_example())
    start, end = encoded["token_spans"]["answer"]
    assert end - start == len("Answer")


def test_layer_numbers_are_transformer_block_indices() -> None:
    result, _ = _extractor().extract_example_with_metadata(_example())
    assert np.all(result[0]["answer"] == 1.0)
    assert np.all(result[1]["answer"] == 2.0)


def test_missing_requested_view_fails_loudly() -> None:
    extractor = _extractor(views=["reasoning"])
    with pytest.raises(ValueError, match="missing requested views"):
        extractor._tokenize_segments(_example())
