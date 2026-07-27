from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a transfer matrix for one pre-specified frozen system"
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--balance_mode", default="balanced")
    parser.add_argument(
        "--metric", default="transfer_tpr_at_1pct_reference_alert_budget_mean"
    )
    args = parser.parse_args()

    report_path = Path(args.results_dir) / "task_frozen_transfer_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing {report_path}. Build the frozen transfer report first.")
    report = pd.read_csv(report_path)
    selected = report[
        (report["probe"].astype(str) == args.probe)
        & (report["k"].astype(int) == args.k)
        & (report["balance_mode"].astype(str) == args.balance_mode)
    ].copy()
    if selected.empty:
        raise ValueError("The requested pre-specified system has no frozen transfer rows")
    key_cols = ["model", "source_task", "target_task"]
    if selected.duplicated(key_cols).any():
        raise ValueError("Frozen report contains multiple rows for the requested system/task cell")

    matrix_rows = []
    for model, model_rows in selected.groupby("model", dropna=False):
        pivot = model_rows.pivot(
            index="source_task",
            columns="target_task",
            values=args.metric,
        )
        for source_task, row in pivot.iterrows():
            for target_task, value in row.items():
                matrix_rows.append(
                    {
                        "model": model,
                        "source_task": source_task,
                        "target_task": target_task,
                        args.metric: value,
                    }
                )
        fig, ax = plt.subplots(figsize=(6, 4))
        values = pivot.to_numpy(dtype=float)
        image = ax.imshow(values, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        ax.set_title(f"{model}: {args.probe}, k={args.k}")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                ax.text(j, i, "nan" if pd.isna(value) else f"{value:.3f}", ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        safe_model = str(model).replace("/", "_")
        fig.savefig(
            Path(args.results_dir) / f"task_transfer_matrix__{safe_model}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    output_path = Path(args.results_dir) / "task_transfer_matrix.csv"
    pd.DataFrame(matrix_rows).to_csv(output_path, index=False)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
