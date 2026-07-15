from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from data.control_features import transform_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Create control feature bundles from existing npz bundles")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--control_type",
        required=True,
        choices=["permute_labels", "permute_features", "gaussian_noise"],
    )
    parser.add_argument(
        "--apply_splits",
        default="train",
        help="Comma-separated splits to corrupt; evaluation labels remain untouched by default.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_splits = {value.strip() for value in args.apply_splits.split(",") if value.strip()}
    for npz_path in sorted(input_dir.glob("*.npz")):
        split = npz_path.name.split("_layer", 1)[0]
        output_path = output_dir / npz_path.name
        if split in apply_splits:
            transform_bundle(npz_path, output_path, args.control_type, seed=args.seed)
        else:
            shutil.copy2(npz_path, output_path)
    print(f"saved control features to {output_dir}")


if __name__ == "__main__":
    main()
