from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


MANIFEST_NAMES = {"submission_manifest.json", "submission_manifest.md"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducibility manifest for a submission bundle")
    parser.add_argument("--bundle_dir", required=True)
    parser.add_argument("--allow_dirty", action="store_true")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Missing bundle directory {bundle_dir}")
    repo_root = Path(__file__).resolve().parents[1]
    code_commit, code_dirty = _git_state(repo_root)
    if code_dirty and not args.allow_dirty:
        raise RuntimeError("Refusing to package a submission from a dirty worktree")

    files = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file() and path.name not in MANIFEST_NAMES:
            files.append(
                {
                    "relative_path": str(path.relative_to(bundle_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    if not files:
        raise ValueError("Refusing to create a manifest for an empty submission bundle")

    provenance_files = []
    for name in ("pyproject.toml", "uv.lock"):
        path = repo_root / name
        if not path.exists():
            raise FileNotFoundError(f"Missing reproducibility file {path}")
        provenance_files.append(
            {"path": name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    manifest = {
        "manifest_schema": "frontier-monitor-submission-v1",
        "bundle_dir": str(bundle_dir.resolve()),
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "num_files": len(files),
        "environment_files": provenance_files,
        "files": files,
    }
    json_path = bundle_dir / "submission_manifest.json"
    markdown_path = bundle_dir / "submission_manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "Submission manifest",
        "",
        f"Code commit: {code_commit}",
        f"Dirty worktree: {code_dirty}",
        f"Files: {len(files)}",
        "",
    ]
    lines.extend(
        f"- {entry['relative_path']} | {entry['size_bytes']} bytes | {entry['sha256']}"
        for entry in files
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {json_path}")
    print(f"saved {markdown_path}")


if __name__ == "__main__":
    main()
