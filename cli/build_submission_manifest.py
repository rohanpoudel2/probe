from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducibility manifest for the submission bundle")
    parser.add_argument("--bundle_dir", required=True)
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    files = []
    for path in sorted(bundle_dir.rglob('*')):
        if path.is_file():
            files.append({
                'relative_path': str(path.relative_to(bundle_dir)),
                'size_bytes': path.stat().st_size,
                'sha256': _sha256(path),
            })

    manifest = {
        'bundle_dir': str(bundle_dir),
        'num_files': len(files),
        'files': files,
    }
    (bundle_dir / 'submission_manifest.json').write_text(json.dumps(manifest, indent=2))
    lines = ['Submission manifest', '', f"Files: {len(files)}", '']
    lines.extend([f"- {f['relative_path']} | {f['size_bytes']} bytes | {f['sha256']}" for f in files])
    (bundle_dir / 'submission_manifest.md').write_text("\n".join(lines))
    print(f"saved {bundle_dir / 'submission_manifest.json'}")
    print(f"saved {bundle_dir / 'submission_manifest.md'}")


if __name__ == '__main__':
    main()
