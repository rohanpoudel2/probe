from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli.common import load_yaml


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _save_dataset(path: Path, dataset) -> None:
    _write_jsonl(path, (dict(row) for row in dataset))
    print(f"saved {path} ({len(dataset)} rows)")


def _fetch_sycophancy_eval(root: Path, cfg: dict) -> None:
    outdir = root / "sycophancy_eval"
    for entry in cfg.get("files", []):
        dataset = load_dataset(
            path=entry.get("loader", "json"),
            data_files=entry["data_files"],
            split=entry.get("split", "train"),
        )
        _save_dataset(outdir / f"{entry['name']}.jsonl", dataset)


def _fetch_motivated_reasoning_raw(root: Path, cfg: dict) -> None:
    outdir = root / "motivated_reasoning_raw"
    for entry in cfg.get("datasets", []):
        for split_alias, split_name in entry.get("splits", {}).items():
            dataset = load_dataset(
                path=entry["repo"],
                name=entry.get("subset"),
                revision=entry.get("revision"),
                split=split_name,
            )
            filename = f"{entry['name']}_{split_alias}.jsonl"
            _save_dataset(outdir / filename, dataset)


def _fetch_honesty_control_raw(root: Path, cfg: dict) -> None:
    dataset_cfg = cfg.get("dataset", {})
    outdir = root / "honesty_control_raw"
    hf_split = dataset_cfg.get("hf_split", "test")
    for split_name in dataset_cfg.get("splits", []):
        dataset = load_dataset(
            path=dataset_cfg["repo"],
            name=split_name,
            revision=dataset_cfg.get("revision"),
            split=hf_split,
        )
        filename = f"mask_{split_name}_{hf_split}.jsonl"
        _save_dataset(outdir / filename, dataset)


def _fetch_cot_monitorability_raw(root: Path, cfg: dict) -> None:
    outdir = root / "cot_monitorability_raw"
    monitorbench = cfg.get("monitorbench", {})
    github_zip = monitorbench.get("github_zip")
    if not github_zip:
        print("skipped cot_monitorability_raw")
        print(f"reason: {monitorbench.get('status', 'no_huggingface_source')}")
        if monitorbench.get("github_repo"):
            print(f"github_repo: {monitorbench['github_repo']}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(github_zip) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(outdir)
    print(f"saved {outdir} (github archive)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the exact upstream Hugging Face sources locked for the paper.")
    parser.add_argument(
        "--source",
        default="all",
        choices=[
            "all",
            "sycophancy_eval",
            "motivated_reasoning_raw",
            "honesty_control_raw",
            "cot_monitorability_raw",
        ],
    )
    parser.add_argument("--lockfile", default="experiments/data/huggingface_source_lock.yaml")
    parser.add_argument("--output_dir", default="data/raw_sources")
    args = parser.parse_args()

    lock = load_yaml(args.lockfile)
    sources = lock.get("sources", {})
    output_dir = Path(args.output_dir)

    fetchers = {
        "sycophancy_eval": _fetch_sycophancy_eval,
        "motivated_reasoning_raw": _fetch_motivated_reasoning_raw,
        "honesty_control_raw": _fetch_honesty_control_raw,
        "cot_monitorability_raw": _fetch_cot_monitorability_raw,
    }

    wanted = fetchers.keys() if args.source == "all" else [args.source]
    for name in wanted:
        fetcher = fetchers[name]
        fetcher(output_dir, sources.get(name, {}))


if __name__ == "__main__":
    main()
