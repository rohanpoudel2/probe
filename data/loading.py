"""Dataset loading and normalisation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata used to keep dataset-specific logic out of the core pipeline."""

    name: str
    task_type: str
    text_field: str
    label_field: str
    positive_label: int
    ood_role: str | None = None
    source: str | None = None


DATASET_SPECS = {
    "enron": DatasetSpec(
        name="enron",
        task_type="spam",
        text_field="text",
        label_field="label",
        positive_label=1,
        source="SetFit/enron_spam",
    ),
    "sms": DatasetSpec(
        name="sms",
        task_type="spam",
        text_field="text",
        label_field="label",
        positive_label=1,
        ood_role="ood_only",
        source="ucirvine/sms_spam",
    ),
    "mask": DatasetSpec(
        name="mask",
        task_type="honesty",
        text_field="text",
        label_field="label",
        positive_label=1,
        source="cais/MASK",
    ),
}


def _load_dataset_source(source: str, split: str | None = None):
    """Load either a HuggingFace dataset id or a local file/directory source."""
    path = Path(source)
    if path.exists():
        if path.is_dir():
            return load_dataset(str(path), split=split)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return load_dataset("json", data_files=str(path), split=split or "train")
        if suffix == ".json":
            return load_dataset("json", data_files=str(path), split=split or "train")
        if suffix == ".csv":
            return load_dataset("csv", data_files=str(path), split=split or "train")
    return load_dataset(source, split=split)


def _concat_splits(ds) -> Dataset:
    if isinstance(ds, Dataset):
        return ds
    all_splits = [ds[split_name] for split_name in ds]
    return concatenate_datasets(all_splits)


def _select_first_present(row: dict[str, Any], candidates: list[str], *, required: bool = False):
    for name in candidates:
        if name in row and row[name] is not None:
            return row[name]
    if required:
        raise KeyError(f"Could not find any of the required fields: {candidates}")
    return None


def _pick_first_existing(example: dict[str, Any], candidates: list[str], default=None):
    for name in candidates:
        if name in example and example[name] is not None:
            return example[name]
    return default


def _normalise_binary_label(value: Any, positive_value: Any = 1) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value == positive_value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        truthy = {"1", "true", "yes", "spam", "positive", "dishonest", "deceptive"}
        falsy = {"0", "false", "no", "ham", "negative", "honest", "truthful"}
        if lowered in truthy:
            return 1
        if lowered in falsy:
            return 0
    return int(value == positive_value)


def load_enron() -> Dataset:
    """Load SetFit/enron_spam and normalise to a common schema."""
    spec = DATASET_SPECS["enron"]
    combined = _concat_splits(load_dataset(spec.source))
    combined = combined.select_columns(["text", "label"])
    combined = combined.add_column("task_type", [spec.task_type] * len(combined))
    return combined


def load_sms() -> Dataset:
    """Load ucirvine/sms_spam and normalise to a common schema."""
    spec = DATASET_SPECS["sms"]
    combined = load_dataset(spec.source)["train"]
    combined = combined.rename_column("sms", "text")
    combined = combined.select_columns(["text", "label"])
    combined = combined.add_column("task_type", [spec.task_type] * len(combined))
    return combined


def _normalize_mask_example(example: dict) -> dict:
    proposition = _pick_first_existing(
        example,
        ["proposition", "question", "prompt", "input", "text"],
        default="",
    )
    ground_truth = _pick_first_existing(
        example,
        ["ground_truth", "answer", "target_answer", "gold_answer", "belief_target"],
        default="",
    )
    pressure_prompt = _pick_first_existing(
        example,
        ["pressure_prompt", "lie_elicitation_prompt", "prompted_text"],
        default="",
    )
    belief_prompt = _pick_first_existing(
        example,
        ["belief_elicitation_prompt", "belief_prompt", "neutral_prompt"],
        default="",
    )
    category = _pick_first_existing(
        example,
        ["category", "archetype", "task_type", "subset"],
        default="unknown",
    )
    question_id = _pick_first_existing(
        example,
        ["question_id", "id", "example_id", "uid"],
        default=None,
    )

    text = pressure_prompt if pressure_prompt else proposition

    raw_label = _pick_first_existing(
        example,
        ["label", "dishonest", "is_dishonest", "target", "y"],
        default=1,
    )
    try:
        label = int(raw_label)
    except Exception:
        label = 1

    return {
        "text": str(text),
        "label": label,
        "belief_target": str(ground_truth),
        "pressure_prompt": str(pressure_prompt),
        "belief_elicitation_prompt": str(belief_prompt),
        "proposition": str(proposition),
        "question_id": question_id,
        "category": str(category),
    }


MASK_CONFIGS = [
    "known_facts",
    "provided_facts",
    "continuations",
    "disinformation",
    "doubling_down_known_facts",
    "statistics",
]


def load_mask(
    split: str | None = None,
    configs: list[str] | None = None,
) -> Dataset:
    """
    Load MASK benchmark and normalize schema.

    Official dataset source:
        cais/MASK

    The HF dataset requires a config name. By default we load all configs
    and concatenate them. Pass configs=["known_facts"] etc. to limit.
    """
    configs = configs or MASK_CONFIGS
    all_datasets = []
    for cfg in configs:
        try:
            ds_cfg = load_dataset("cais/MASK", cfg)
        except Exception as e:
            print(f"  Warning: could not load MASK config '{cfg}': {e}")
            continue
        combined = _concat_splits(ds_cfg)
        all_datasets.append(combined)

    if not all_datasets:
        raise RuntimeError("Could not load any MASK configs")

    raw_combined = concatenate_datasets(all_datasets)

    if split is not None:
        raise ValueError(
            f"split={split} not supported when loading multiple MASK configs. "
            "The concatenated dataset has no named splits."
        )

    ds = raw_combined

    ds = ds.map(_normalize_mask_example)

    keep_cols = [
        "text",
        "label",
        "belief_target",
        "pressure_prompt",
        "belief_elicitation_prompt",
        "proposition",
        "question_id",
        "category",
    ]
    existing = [c for c in keep_cols if c in ds.column_names]
    ds = ds.select_columns(existing)
    return ds
