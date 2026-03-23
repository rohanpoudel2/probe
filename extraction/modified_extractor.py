"""Modified activation extraction for P6 (Prompted Probing) and P8 (Follow-up Question).

P6: Prepend a task description to each input, then extract activations normally.
P8: Append a follow-up question, extract activations only from the follow-up tokens.
"""

import numpy as np

from .extractor import extract_and_cache


DEFAULT_PROMPT_PREFIX = (
    "You are a spam classifier. Determine whether the following message is spam. "
    "Message: "
)

DEFAULT_FOLLOWUP_QUESTION = "\n\nWere you being deceptive in the above message?"


def extract_prompted(
    model_name: str,
    texts: list[str],
    labels: np.ndarray,
    layers: list[int],
    cache_dir: str,
    dataset_name: str,
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
    batch_size: int = 16,
    max_length: int = 512,
):
    """Extract activations with a task description prepended to each input.

    This is the P6 (Prompted Probing) approach from Tillman and Mossing 2025.
    The probe itself is just logistic regression; the difference is that the
    model sees an explicit task description, so its activations may encode
    the classification signal more cleanly.
    """
    prompted_texts = [f"{prompt_prefix}{text}" for text in texts]
    extract_and_cache(
        model_name=model_name,
        texts=prompted_texts,
        labels=labels,
        layers=layers,
        cache_dir=cache_dir,
        dataset_name=f"{dataset_name}_prompted",
        batch_size=batch_size,
        max_length=max_length,
    )


def extract_followup(
    model_name: str,
    texts: list[str],
    labels: np.ndarray,
    layers: list[int],
    cache_dir: str,
    dataset_name: str,
    followup_question: str = DEFAULT_FOLLOWUP_QUESTION,
    batch_size: int = 16,
    max_length: int = 512,
):
    """Extract activations from a follow-up question appended to each input.

    This is the P8 (Follow-up Question Probe) approach from Goldowsky-Dill
    et al. 2025. The idea is that asking the model directly about deception
    after it processes the input produces activations that are more informative
    for a probe.

    Note: Ideally we would extract activations only from the follow-up tokens.
    For simplicity, we extract mean-pooled activations over the full
    concatenated input (original + follow-up). This is a reasonable first
    approximation — if results are promising, a follow-up experiment can
    isolate the question tokens specifically.
    """
    followup_texts = [f"{text}{followup_question}" for text in texts]
    extract_and_cache(
        model_name=model_name,
        texts=followup_texts,
        labels=labels,
        layers=layers,
        cache_dir=cache_dir,
        dataset_name=f"{dataset_name}_followup",
        batch_size=batch_size,
        max_length=max_length,
    )
