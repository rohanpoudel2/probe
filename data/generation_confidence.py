from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np

from data.rollout_schema import content_hash


GENERATION_CONFIDENCE_SCHEMA_VERSION = "generation-confidence-v1"
CONFIDENCE_FEATURE_NAMES = (
    "log1p_response_tokens",
    "selected_logprob_mean",
    "selected_logprob_std",
    "selected_logprob_min",
    "selected_logprob_max",
    "selected_logprob_q10",
    "selected_logprob_q25",
    "selected_logprob_median",
    "selected_logprob_q75",
    "selected_logprob_q90",
    "selected_logprob_sum_over_sqrt_n",
    "selected_logprob_first_quarter_mean",
    "selected_logprob_last_quarter_mean",
    "entropy_mean",
    "entropy_std",
    "entropy_max",
    "entropy_q90",
    "top1_top2_probability_margin_mean",
    "top1_top2_probability_margin_min",
    "top1_top2_probability_margin_q10",
    "selected_token_top1_rate",
    "final_token_logprob",
)


def _trace_payload(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        key: trace[key]
        for key in (
            "schema_version",
            "score_semantics",
            "response_token_ids_sha256",
            "selected_token_logprobs",
            "token_entropies",
            "top1_top2_probability_margins",
            "selected_token_is_top1",
        )
    }


def build_generation_confidence_trace(
    step_scores: Iterable[Any], response_token_ids: Any
) -> dict[str, Any]:
    """Summarize the exact processed distribution used at each generation step."""

    import torch

    token_ids = torch.as_tensor(response_token_ids, dtype=torch.long).reshape(-1)
    scores = list(step_scores)
    if len(scores) != len(token_ids) or not len(scores):
        raise ValueError(
            "Generation scores must align one-to-one with non-empty response token IDs"
        )
    selected_logprobs: list[float] = []
    entropies: list[float] = []
    margins: list[float] = []
    selected_is_top1: list[bool] = []
    token_id_list = token_ids.tolist()
    # Summaries are computed in batches over the step-score matrix. Every stored
    # value (selected log-prob, entropy, top-1/top-2 margin, is-top-1) is
    # identical to a per-token computation up to floating-point reduction order;
    # batching collapses the per-token Python/kernel dispatch that dominates
    # wall-clock on MPS. The chunk bounds peak memory for large vocabularies.
    chunk_size = 64
    for start in range(0, len(scores), chunk_size):
        chunk_scores = scores[start : start + chunk_size]
        chunk_token_ids = token_id_list[start : start + chunk_size]
        rows: list[Any] = []
        for offset, score in enumerate(chunk_scores):
            values = torch.as_tensor(score).detach().float()
            if values.ndim == 2 and values.shape[0] == 1:
                values = values[0]
            if values.ndim != 1 or not 0 <= int(chunk_token_ids[offset]) < len(values):
                raise ValueError(
                    f"Invalid generation score tensor at response token {start + offset}"
                )
            rows.append(values)
        stacked = torch.stack(rows, dim=0)
        log_probs = torch.log_softmax(stacked, dim=-1)
        probabilities = torch.exp(log_probs)
        entropy_terms = torch.where(
            probabilities > 0,
            probabilities * torch.nan_to_num(log_probs, neginf=0.0),
            torch.zeros_like(probabilities),
        )
        chunk_entropy = -entropy_terms.sum(dim=-1)
        top_probabilities, top_indices = torch.topk(probabilities, k=2, dim=-1)
        chunk_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
        selection = torch.as_tensor(
            chunk_token_ids, dtype=torch.long, device=stacked.device
        )
        row_index = torch.arange(stacked.shape[0], device=stacked.device)
        chunk_selected = log_probs[row_index, selection]
        chunk_is_top1 = top_indices[:, 0] == selection
        selected_logprobs.extend(float(value) for value in chunk_selected.tolist())
        entropies.extend(float(value) for value in chunk_entropy.tolist())
        margins.extend(float(value) for value in chunk_margin.tolist())
        selected_is_top1.extend(bool(value) for value in chunk_is_top1.tolist())

    trace: dict[str, Any] = {
        "schema_version": GENERATION_CONFIDENCE_SCHEMA_VERSION,
        "score_semantics": "generation_processed_scores_normalized_over_vocabulary",
        "response_token_ids_sha256": content_hash(token_ids.tolist()),
        "selected_token_logprobs": selected_logprobs,
        "token_entropies": entropies,
        "top1_top2_probability_margins": margins,
        "selected_token_is_top1": selected_is_top1,
    }
    trace["trace_sha256"] = content_hash(_trace_payload(trace))
    validate_generation_confidence_trace(
        trace,
        expected_token_count=len(token_ids),
        expected_token_ids=token_ids.tolist(),
    )
    return trace


def validate_generation_confidence_trace(
    trace: Any,
    *,
    expected_token_count: int | None = None,
    expected_token_ids: list[int] | None = None,
) -> dict[str, Any]:
    if not isinstance(trace, dict):
        raise ValueError("generation.confidence_trace must be an object")
    if trace.get("schema_version") != GENERATION_CONFIDENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported generation confidence schema")
    if trace.get("score_semantics") != (
        "generation_processed_scores_normalized_over_vocabulary"
    ):
        raise ValueError("Unsupported generation confidence score semantics")
    token_ids_hash = str(trace.get("response_token_ids_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", token_ids_hash) is None:
        raise ValueError("Generation confidence lacks a valid response-token hash")
    if expected_token_ids is not None and token_ids_hash != content_hash(
        expected_token_ids
    ):
        raise ValueError("Generation confidence does not match response token IDs")
    arrays: dict[str, list[Any]] = {}
    for key in (
        "selected_token_logprobs",
        "token_entropies",
        "top1_top2_probability_margins",
        "selected_token_is_top1",
    ):
        value = trace.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"generation confidence trace requires non-empty {key}")
        arrays[key] = value
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Generation confidence arrays must have equal lengths")
    observed_count = lengths.pop()
    if expected_token_count is not None and observed_count != expected_token_count:
        raise ValueError(
            f"Generation confidence has {observed_count} tokens; expected {expected_token_count}"
        )
    for key in (
        "selected_token_logprobs",
        "token_entropies",
        "top1_top2_probability_margins",
    ):
        values = np.asarray(arrays[key], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Generation confidence {key} must be finite")
    if np.any(np.asarray(arrays["selected_token_logprobs"], dtype=float) > 1e-6):
        raise ValueError("Selected-token log probabilities cannot be positive")
    if np.any(np.asarray(arrays["token_entropies"], dtype=float) < -1e-6):
        raise ValueError("Token entropies cannot be negative")
    probability_margins = np.asarray(
        arrays["top1_top2_probability_margins"], dtype=float
    )
    if np.any((probability_margins < -1e-6) | (probability_margins > 1.0 + 1e-6)):
        raise ValueError("Top-1/2 probability margins must lie in [0, 1]")
    if any(not isinstance(value, bool) for value in arrays["selected_token_is_top1"]):
        raise ValueError("selected_token_is_top1 must contain booleans")
    if trace.get("trace_sha256") != content_hash(_trace_payload(trace)):
        raise ValueError("Generation confidence trace hash mismatch")
    return trace


def confidence_feature_vector(generation: Any) -> np.ndarray:
    if not isinstance(generation, dict):
        raise ValueError("Labeled rollout lacks generation metadata")
    token_ids = generation.get("response_token_ids")
    token_count = generation.get("response_token_count")
    if not isinstance(token_ids, list) or not isinstance(token_count, int):
        raise ValueError("Generation metadata lacks response token IDs/count")
    if token_count != len(token_ids) or token_count < 1:
        raise ValueError("Generation response token IDs/count are inconsistent")
    trace = validate_generation_confidence_trace(
        generation.get("confidence_trace"),
        expected_token_count=token_count,
        expected_token_ids=token_ids,
    )
    logprobs = np.asarray(trace["selected_token_logprobs"], dtype=float)
    entropies = np.asarray(trace["token_entropies"], dtype=float)
    margins = np.asarray(trace["top1_top2_probability_margins"], dtype=float)
    top1 = np.asarray(trace["selected_token_is_top1"], dtype=float)
    quarter = max(1, int(math.ceil(token_count / 4)))
    values = np.asarray(
        [
            math.log1p(token_count),
            np.mean(logprobs),
            np.std(logprobs),
            np.min(logprobs),
            np.max(logprobs),
            np.quantile(logprobs, 0.10),
            np.quantile(logprobs, 0.25),
            np.median(logprobs),
            np.quantile(logprobs, 0.75),
            np.quantile(logprobs, 0.90),
            np.sum(logprobs) / math.sqrt(token_count),
            np.mean(logprobs[:quarter]),
            np.mean(logprobs[-quarter:]),
            np.mean(entropies),
            np.std(entropies),
            np.max(entropies),
            np.quantile(entropies, 0.90),
            np.mean(margins),
            np.min(margins),
            np.quantile(margins, 0.10),
            np.mean(top1),
            logprobs[-1],
        ],
        dtype=np.float64,
    )
    if len(values) != len(CONFIDENCE_FEATURE_NAMES) or not np.all(np.isfinite(values)):
        raise ValueError(
            "Generation confidence produced invalid fixed-dimensional features"
        )
    return values
