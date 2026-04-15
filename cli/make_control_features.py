from __future__ import annotations

import argparse
from pathlib import Path

from data.control_features import transform_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Create control feature bundles from existing npz bundles")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--control_type", required=True, choices=["permute_labels", "shuffle_rows", "gaussian_noise"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    for npz_path in sorted(input_dir.glob("*.npz")):
        transform_bundle(npz_path, output_dir / npz_path.name, args.control_type, seed=args.seed)
    print(f"saved control features to {output_dir}")


if __name__ == "__main__":
    main()
