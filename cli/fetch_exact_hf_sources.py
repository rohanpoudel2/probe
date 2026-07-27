from __future__ import annotations

import argparse
import hashlib
import io
import json
import heapq
import os
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterable

from datasets import load_dataset

from cli.common import load_yaml
from data.monitorbench import (
    create_monitorbench_source_manifest,
    load_monitorbench_adapter,
    validate_monitorbench_source_manifest,
)


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _require_commit_revision(revision: str | None, source: str) -> str:
    if not revision or not COMMIT_RE.fullmatch(revision):
        raise ValueError(f"{source} must be pinned to a full 40-character commit hash")
    return revision


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")


def _save_dataset(path: Path, dataset) -> None:
    _write_jsonl(path, (dict(row) for row in dataset))
    print(f"saved {path} ({len(dataset)} rows)")


def _fetch_sycophancy_eval(root: Path, cfg: dict) -> None:
    _require_commit_revision(cfg.get("revision"), "sycophancy_eval")
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
            revision = _require_commit_revision(entry.get("revision"), entry["repo"])
            dataset = load_dataset(
                path=entry["repo"],
                name=entry.get("subset"),
                revision=revision,
                split=split_name,
            )
            filename = f"{entry['name']}_{split_alias}.jsonl"
            _save_dataset(outdir / filename, dataset)


def _fetch_honesty_control_raw(root: Path, cfg: dict) -> None:
    dataset_cfg = cfg.get("dataset", {})
    outdir = root / "honesty_control_raw"
    hf_split = dataset_cfg.get("hf_split", "test")
    revision = _require_commit_revision(
        dataset_cfg.get("revision"), dataset_cfg.get("repo", "MASK")
    )
    for split_name in dataset_cfg.get("splits", []):
        dataset = load_dataset(
            path=dataset_cfg["repo"],
            name=split_name,
            revision=revision,
            split=hf_split,
        )
        filename = f"mask_{split_name}_{hf_split}.jsonl"
        _save_dataset(outdir / filename, dataset)


def _fetch_reference_traffic_raw(root: Path, cfg: dict) -> None:
    """Stream a reproducible content-hash sample without downloading the full corpus."""

    dataset_cfg = cfg.get("dataset", {})
    revision = _require_commit_revision(
        dataset_cfg.get("revision"),
        dataset_cfg.get("repo", "reference traffic source"),
    )
    max_rows = int(dataset_cfg.get("raw_candidate_rows", 50_000))
    if max_rows < 1:
        raise ValueError("reference_traffic_raw.raw_candidate_rows must be positive")
    dataset = load_dataset(
        path=dataset_cfg["repo"],
        revision=revision,
        split=dataset_cfg.get("split", "train"),
        streaming=True,
    )
    # Keep the lexicographically smallest SHA-256 ranks. This is deterministic
    # for the pinned source and independent of streaming shard order.
    heap: list[tuple[int, str, int, dict]] = []
    for index, raw in enumerate(dataset):
        row = dict(raw)
        stable_key = str(
            row.get(dataset_cfg.get("sample_key", "conversation_hash")) or index
        )
        rank_hex = hashlib.sha256(
            f"{dataset_cfg.get('sample_seed', 42)}:{stable_key}".encode("utf-8")
        ).hexdigest()
        rank = int(rank_hex, 16)
        item = (-rank, stable_key, index, row)
        if len(heap) < max_rows:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)
    selected = [
        item[3] for item in sorted(heap, key=lambda item: (-item[0], item[1], item[2]))
    ]
    if not selected:
        raise ValueError("Pinned reference-traffic source returned no rows")
    _write_jsonl(
        root / "reference_traffic_raw" / "wildchat_train_sample.jsonl", selected
    )
    print(
        f"saved {root / 'reference_traffic_raw' / 'wildchat_train_sample.jsonl'} "
        f"({len(selected)} deterministic sampled rows)"
    )


def _fetch_cot_monitorability_raw(root: Path, cfg: dict) -> None:
    outdir = root / "cot_monitorability_raw"
    monitorbench = cfg.get("monitorbench", {})
    adapter_path = Path(
        monitorbench.get(
            "adapter_contract", "experiments/protocol/monitorbench_adapter.yaml"
        )
    )
    adapter, _ = load_monitorbench_adapter(adapter_path)
    adapter_source = adapter["source"]
    locked_values = {
        "github_repo": adapter_source["repository"],
        "revision": adapter_source["revision"],
        "github_zip": adapter_source["archive_url"],
        "sha256": adapter_source["archive_sha256"],
    }
    mismatches = {
        key: (monitorbench.get(key), expected)
        for key, expected in locked_values.items()
        if monitorbench.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"MonitorBench source lock differs from the adapter contract: {mismatches}"
        )
    manifest_path = outdir / "monitorbench_source_manifest.json"
    if manifest_path.exists():
        validate_monitorbench_source_manifest(manifest_path, adapter=adapter)
        print(f"verified existing {manifest_path}")
        return
    github_zip = monitorbench.get("github_zip")
    if not github_zip:
        print("skipped cot_monitorability_raw")
        print(f"reason: {monitorbench.get('status', 'no_huggingface_source')}")
        if monitorbench.get("github_repo"):
            print(f"github_repo: {monitorbench['github_repo']}")
        return

    with urllib.request.urlopen(github_zip) as response:
        payload = response.read()
    expected_sha256 = monitorbench.get("sha256")
    if not expected_sha256:
        raise ValueError("MonitorBench archive must have a locked sha256")
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"MonitorBench archive checksum mismatch: expected {expected_sha256}, got {observed_sha256}"
        )
    _install_monitorbench_archive(
        payload=payload,
        outdir=outdir,
        adapter=adapter,
    )
    print(f"saved and verified {manifest_path}")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _safe_monitorbench_members(
    archive: zipfile.ZipFile,
) -> tuple[str, list[tuple[zipfile.ZipInfo, PurePosixPath]]]:
    members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    roots: set[str] = set()
    for member in archive.infolist():
        if "\\" in member.filename:
            raise ValueError(
                f"Unsafe backslash path in MonitorBench archive: {member.filename}"
            )
        path = PurePosixPath(member.filename)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError(f"Unsafe path in MonitorBench archive: {member.filename}")
        roots.add(path.parts[0])
        unix_mode = member.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise ValueError(
                f"Symlink prohibited in MonitorBench archive: {member.filename}"
            )
        file_type = stat.S_IFMT(unix_mode)
        if not member.is_dir() and file_type not in {0, stat.S_IFREG}:
            raise ValueError(
                f"Non-regular file prohibited in MonitorBench archive: {member.filename}"
            )
        members.append((member, path))
    if len(roots) != 1 or not members:
        raise ValueError("MonitorBench archive must contain one non-empty root directory")
    return roots.pop(), members


def _install_monitorbench_archive(
    *, payload: bytes, outdir: Path, adapter: dict
) -> Path:
    source = adapter["source"]
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != source["archive_sha256"]:
        raise ValueError(
            "MonitorBench archive checksum differs from the adapter contract"
        )
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "monitorbench_source_manifest.json"
    if manifest_path.exists():
        validate_monitorbench_source_manifest(manifest_path, adapter=adapter)
        return manifest_path

    revision = source["revision"]
    archive_path = outdir / f"monitorbench-{revision}.zip"
    destination_root = outdir / revision
    if destination_root.exists():
        raise FileExistsError(
            f"Refusing to mix MonitorBench source into unmanaged path {destination_root}"
        )
    _atomic_write_bytes(archive_path, payload)

    with tempfile.TemporaryDirectory(prefix=".monitorbench-", dir=outdir) as temporary:
        staging_root = Path(temporary) / revision
        staging_root.mkdir()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive_root, members = _safe_monitorbench_members(archive)
            expected_root = f"MonitorBench-{revision}"
            if archive_root != expected_root:
                raise ValueError(
                    f"MonitorBench archive root is {archive_root!r}, expected {expected_root!r}"
                )
            for member, member_path in members:
                relative_parts = member_path.parts[1:]
                if not relative_parts:
                    continue
                target = staging_root.joinpath(*relative_parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source_handle, target.open("wb") as output:
                    shutil.copyfileobj(source_handle, output)
        for relative_path, expected_digest in source["critical_files"].items():
            critical_path = staging_root / relative_path
            if (
                not critical_path.is_file()
                or hashlib.sha256(critical_path.read_bytes()).hexdigest()
                != expected_digest
            ):
                raise ValueError(
                    f"MonitorBench critical source file mismatch: {relative_path}"
                )
        os.replace(staging_root, destination_root)

    manifest = create_monitorbench_source_manifest(
        manifest_path=manifest_path,
        archive_path=archive_path,
        extracted_root=destination_root,
        adapter=adapter,
    )
    _atomic_write_json(manifest_path, manifest)
    validate_monitorbench_source_manifest(manifest_path, adapter=adapter)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the exact upstream sources locked for the registered study."
    )
    parser.add_argument(
        "--source",
        default="all",
        choices=[
            "all",
            "sycophancy_eval",
            "motivated_reasoning_raw",
            "honesty_control_raw",
            "reference_traffic_raw",
            "cot_monitorability_raw",
        ],
    )
    parser.add_argument(
        "--lockfile", default="experiments/data/huggingface_source_lock.yaml"
    )
    parser.add_argument("--output_dir", default="data/raw_sources")
    args = parser.parse_args()

    lock = load_yaml(args.lockfile)
    sources = lock.get("sources", {})
    output_dir = Path(args.output_dir)

    fetchers = {
        "sycophancy_eval": _fetch_sycophancy_eval,
        "motivated_reasoning_raw": _fetch_motivated_reasoning_raw,
        "honesty_control_raw": _fetch_honesty_control_raw,
        "reference_traffic_raw": _fetch_reference_traffic_raw,
        "cot_monitorability_raw": _fetch_cot_monitorability_raw,
    }

    wanted = fetchers.keys() if args.source == "all" else [args.source]
    for name in wanted:
        fetcher = fetchers[name]
        fetcher(output_dir, sources.get(name, {}))


if __name__ == "__main__":
    main()
