"""Load Enron-Spam and SMS Spam datasets from HuggingFace."""

from datasets import load_dataset, Dataset


def load_enron() -> Dataset:
    """Load SetFit/enron_spam, normalise to {text, label} where 1=spam."""
    ds = load_dataset("SetFit/enron_spam")
    # This dataset has train/test splits with 'text' and 'label' columns
    # label: 1 = spam, 0 = ham (already correct)
    # Concatenate all splits into one pool — we do our own splitting
    all_splits = []
    for split_name in ds:
        all_splits.append(ds[split_name])

    from datasets import concatenate_datasets
    combined = concatenate_datasets(all_splits)

    # Keep only the columns we need
    combined = combined.select_columns(["text", "label"])
    return combined


def load_sms() -> Dataset:
    """Load ucirvine/sms_spam, normalise to {text, label} where 1=spam."""
    ds = load_dataset("ucirvine/sms_spam")
    # This dataset has a 'train' split with 'sms' and 'label' columns
    # label: 1 = spam, 0 = ham (already correct)
    combined = ds["train"]

    # Rename 'sms' -> 'text' for consistency
    combined = combined.rename_column("sms", "text")
    combined = combined.select_columns(["text", "label"])
    return combined
