"""Filter dataset samples by model's internal confidence (logit difference).

Following the base paper protocol: only keep samples where the model's logit
difference between "Yes" and "No" tokens signals a confident internal belief.
This isolates the probe's ability to detect the target concept from confounds
related to factual uncertainty.
"""

import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset


def _get_yes_no_token_ids(tokenizer):
    """Get token IDs for 'Yes' and 'No' across different tokenizers."""
    # Try common variations
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


def _format_prompt(text: str) -> str:
    """Format a text sample as a yes/no classification question."""
    return f"Is the following message spam? Answer Yes or No.\n\n{text}\n\nAnswer:"


@torch.no_grad()
def filter_by_logit_confidence(
    dataset: Dataset,
    model,
    tokenizer,
    threshold: float = 0.0,
    batch_size: int = 8,
    max_length: int = 512,
) -> tuple[Dataset, np.ndarray]:
    """Filter dataset to samples where the model has confident internal belief.

    Args:
        dataset: Dataset with 'text' and 'label' columns.
        model: A causal LM (already on device, eval mode).
        tokenizer: Matching tokenizer.
        threshold: Minimum absolute logit difference to keep a sample.
        batch_size: Batch size for forward passes.
        max_length: Max token length for inputs.

    Returns:
        Filtered dataset and array of logit differences for kept samples.
    """
    model.eval()
    device = next(model.parameters()).device
    yes_id, no_id = _get_yes_no_token_ids(tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logit_diffs = []
    prompts = [_format_prompt(t) for t in dataset["text"]]

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
        # Get logits at the last non-padding token for each sample
        for j in range(len(batch_prompts)):
            attention_mask = inputs["attention_mask"][j]
            last_idx = attention_mask.sum() - 1
            logits = outputs.logits[j, last_idx]
            diff = (logits[yes_id] - logits[no_id]).item()
            logit_diffs.append(diff)

    logit_diffs = np.array(logit_diffs)

    # For spam (label=1), we expect model to lean "Yes" (positive diff)
    # For ham (label=0), we expect model to lean "No" (negative diff)
    # "Confident" means the model's belief aligns with the ground truth
    aligned_diffs = np.where(
        np.array(dataset["label"]) == 1, logit_diffs, -logit_diffs
    )
    keep_mask = aligned_diffs > threshold
    keep_indices = np.where(keep_mask)[0]

    filtered = dataset.select(keep_indices.tolist())
    return filtered, logit_diffs[keep_indices]


def save_filtered_indices(indices: np.ndarray, path: Path):
    """Save filtered sample indices for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(indices.tolist(), f)


def load_filtered_indices(path: Path) -> np.ndarray:
    """Load previously saved filtered indices."""
    with open(path) as f:
        return np.array(json.load(f))
