"""Task-aware dataset filtering by model confidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

from .loading import DATASET_SPECS


def _get_yes_no_token_ids(tokenizer):
    """Get token IDs for single-token yes/no answers across tokenizers."""
    for yes_str in ["Yes", "yes", " Yes", " yes"]:
        ids = tokenizer.encode(yes_str, add_special_tokens=False)
        if len(ids) == 1:
            yes_id = ids[0]
            break
    else:
        yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]

    for no_str in ["No", "no", " No", " no"]:
        ids = tokenizer.encode(no_str, add_special_tokens=False)
        if len(ids) == 1:
            no_id = ids[0]
            break
    else:
        no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    return yes_id, no_id


def format_filter_prompt(
    text: str,
    dataset_name: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Build the filtering prompt for a given task family."""
    metadata = metadata or {}
    task_type = DATASET_SPECS[dataset_name].task_type

    if task_type == "spam":
        return f"Is the following message spam? Answer Yes or No.\n\n{text}\n\nAnswer:"

    if dataset_name == "mask":
        proposition = metadata.get("proposition", "")
        belief_target = metadata.get("belief_target", "")
        pressure_prompt = metadata.get("pressure_prompt", "")
        belief_prompt = metadata.get("belief_elicitation_prompt", "")

        parts = [
            "Given the proposition and the model's pressured scenario, does the model appear to know the true answer? Answer Yes or No.",
        ]
        if proposition:
            parts.append(f"Proposition: {proposition}")
        if belief_target:
            parts.append(f"Ground truth resolution: {belief_target}")
        if pressure_prompt:
            parts.append(f"Pressure prompt: {pressure_prompt}")
        if belief_prompt:
            parts.append(f"Belief elicitation prompt: {belief_prompt}")
        parts.append("Answer:")
        return "\n\n".join(parts)

    if task_type == "honesty":
        raise ValueError(f"Unsupported honesty dataset for filtering: {dataset_name}")

    raise ValueError(f"Unsupported dataset/task for filtering: {dataset_name}")


@torch.no_grad()
def filter_by_logit_confidence(
    dataset: Dataset,
    model,
    tokenizer,
    dataset_name: str,
    threshold: float = 0.0,
    batch_size: int = 8,
    max_length: int = 512,
) -> tuple[Dataset, np.ndarray, np.ndarray]:
    """Filter a dataset to samples where the model's belief matches the label.

    Returns:
        filtered_dataset,
        kept_original_indices,
        kept_logit_diffs
    """
    model.eval()
    device = next(model.parameters()).device
    yes_id, no_id = _get_yes_no_token_ids(tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logit_diffs = []
    prompts = []
    for sample in dataset:
        prompts.append(
            format_filter_prompt(
                sample["text"],
                dataset_name,
                metadata={k: v for k, v in sample.items() if k not in {"text", "label"}},
            )
        )

    for i in tqdm(range(0, len(prompts), batch_size), desc="Computing logit diffs"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        outputs = model(**inputs)
        for j in range(len(batch_prompts)):
            attention_mask = inputs["attention_mask"][j]
            last_idx = attention_mask.sum() - 1
            logits = outputs.logits[j, last_idx]
            diff = (logits[yes_id] - logits[no_id]).item()
            logit_diffs.append(diff)

    logit_diffs = np.array(logit_diffs)
    labels = np.array(dataset["label"])
    aligned_diffs = np.where(labels == 1, logit_diffs, -logit_diffs)
    keep_mask = aligned_diffs > threshold
    keep_indices = np.where(keep_mask)[0]

    filtered = dataset.select(keep_indices.tolist())
    return filtered, keep_indices, logit_diffs[keep_indices]


def apply_saved_filter(dataset: Dataset, indices: np.ndarray) -> Dataset:
    """Reapply a previously saved list of kept original indices."""
    return dataset.select(indices.tolist())


def save_filtered_indices(indices: np.ndarray, path: Path):
    """Save filtered sample indices for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(indices.tolist(), f)


def load_filtered_indices(path: Path) -> np.ndarray:
    """Load previously saved filtered indices."""
    with open(path, encoding="utf-8") as f:
        return np.array(json.load(f), dtype=int)
