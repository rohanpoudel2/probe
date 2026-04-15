from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _mean_metric(results_dir: Path, metric: str) -> float:
    path = results_dir / "task_cross_task_transfer.csv"
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path)
    if df.empty:
        return float("nan")
    return float(df[metric].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare main results to negative control suites")
    parser.add_argument("--main_results_dir", required=True)
    parser.add_argument("--controls_root", required=True)
    parser.add_argument("--metric", default="transfer_recall_at_1pct_fpr_mean")
    args = parser.parse_args()

    main_results_dir = Path(args.main_results_dir)
    controls_root = Path(args.controls_root)
    main_mean = _mean_metric(main_results_dir, args.metric)

    rows = []
    for subdir in sorted(controls_root.iterdir()):
        if not subdir.is_dir():
            continue
        ctrl_mean = _mean_metric(subdir, args.metric)
        rows.append({
            "control_name": subdir.name,
            "main_mean": main_mean,
            "control_mean": ctrl_mean,
            "main_minus_control": main_mean - ctrl_mean,
            "metric": args.metric,
        })
    out = pd.DataFrame(rows)
    out_path = main_results_dir / "negative_control_report.csv"
    out.to_csv(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
