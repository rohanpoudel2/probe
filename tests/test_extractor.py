from dataclasses import replace
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
    extractor.resolved_device = "cpu"
    extractor.model_parameter_device = "cpu"
    extractor.model_parameter_dtype = "torch.float32"
    extractor.chat_template_sha256 = "f" * 64
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


def test_split_extraction_resumes_from_atomic_chunk_checkpoints(
    tmp_path, monkeypatch
) -> None:
    output_dir = tmp_path / "features"
    extractor = _extractor(
        output_dir=str(output_dir),
        dataset_sha256="d" * 64,
        code_revision="c" * 40,
    )
    first = replace(_example(), task_family="reference_traffic")
    second = replace(first, example_id="r2", question_id="q2")

    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(
        "extraction.task_extractor.time.monotonic", lambda: next(ticks)
    )
    assert not extractor.extract_split(
        [first, second],
        "train",
        checkpoint_examples=1,
        deadline_monotonic=1.0,
    )

    resumed = _extractor(
        output_dir=str(output_dir),
        dataset_sha256="d" * 64,
        code_revision="c" * 40,
    )
    original_extract = resumed.extract_example_with_metadata
    recomputed_ids = []

    def record_extract(example):
        recomputed_ids.append(example.example_id)
        return original_extract(example)

    monkeypatch.setattr(resumed, "extract_example_with_metadata", record_extract)
    assert resumed.extract_split(
        [first, second], "train", checkpoint_examples=1
    )
    assert recomputed_ids == ["r2"]
    with np.load(output_dir / "train_layer0.npz", allow_pickle=False) as bundle:
        assert bundle["example_ids"].astype(str).tolist() == ["r1", "r2"]
