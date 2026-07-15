from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cli.common import load_yaml, run_cmd
from tasks import TASK_REGISTRY


def _iter_pairs(cfg: dict) -> list[dict]:
    pairs = []
    for pair in cfg.get("calibration_pairs", []):
        pairs.append({**pair, "pair_type": "calibration"})
    for pair in cfg.get("transfer_pairs", []):
        pairs.append({**pair, "pair_type": "transfer"})
    for pair in cfg.get("task_pairs", []):
        pairs.append({**pair, "pair_type": "unspecified"})
    return pairs


def _csv(value) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _baseline_available(task_name: str, baseline: str) -> bool:
    task_cls = TASK_REGISTRY.get(task_name)
    return task_cls is None or baseline not in task_cls().spec.unavailable_baselines


def _cfg_device(cfg: dict, key: str, fallback: str = "auto") -> str:
    return str(cfg.get(key, fallback))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run protocol-aware multi-model benchmark from YAML config"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional default execution device forwarded to model-backed baselines",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    judge_device = (
        args.device if args.device is not None else _cfg_device(cfg, "llm_judge_device")
    )
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    overwritten_pairs: set[tuple[str, str]] = set()
    overwritten_baselines: set[tuple[str, str, str]] = set()
    run_black_box = bool(cfg.get("run_black_box_baselines", False))
    registered_baselines = set(cfg.get("black_box_baselines", []))
    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        feature_dirs = model_cfg["feature_dirs"]
        calibration_dirs = model_cfg.get("calibration_dirs", {})
        for pair in _iter_pairs(cfg):
            source_task = pair["source_task"]
            target_task = pair["target_task"]
            cmd = [
                sys.executable,
                "-m",
                "cli.run_task_sweep",
                "--source_dir",
                feature_dirs[source_task],
                "--source_task",
                source_task,
                "--calibration_dir",
                calibration_dirs.get(source_task, feature_dirs[source_task]),
                "--target_dir",
                feature_dirs[target_task],
                "--target_task",
                target_task,
                "--model",
                model_name,
                "--results_dir",
                str(results_dir),
                "--views",
                cfg.get("views", "full_text,answer"),
                "--layers",
                str(cfg.get("layers", "all")),
                "--probes",
                cfg.get(
                    "probes", "P1_logistic,P2_mass_mean,P3_lda,P4_cosine,P7_mahalanobis"
                ),
                "--k_values",
                cfg.get("k_values", "1,2,4,8"),
                "--seeds",
                str(cfg.get("seeds", 5)),
                "--balance_modes",
                cfg.get("balance_modes", "balanced,imbalanced"),
                "--max_fpr",
                str(cfg.get("max_fpr", 0.01)),
                "--min_calibration_negatives",
                str(cfg.get("min_calibration_negatives", 1000)),
            ]
            pair_key = (source_task, target_task)
            if cfg.get("overwrite", False) and pair_key not in overwritten_pairs:
                cmd.append("--overwrite")
                overwritten_pairs.add(pair_key)
            run_cmd(cmd)

            if not run_black_box:
                continue
            labeled_data = model_cfg["labeled_data"]
            benign_data = model_cfg["benign_labeled_data"]
            text_views = _csv(cfg["text_embedding_views"])
            if "B1_text_tfidf" in registered_baselines:
                baseline_key = ("B1_text_tfidf", source_task, target_task)
                text_cmd = [
                    sys.executable,
                    "-m",
                    "cli.run_text_baselines",
                    "--source_task",
                    source_task,
                    "--source_data",
                    labeled_data[source_task],
                    "--target_task",
                    target_task,
                    "--target_data",
                    labeled_data[target_task],
                    "--calibration_task",
                    "benign_calibration",
                    "--calibration_data",
                    benign_data,
                    "--model",
                    model_name,
                    "--results_dir",
                    str(results_dir),
                    "--views",
                    text_views,
                    "--k_values",
                    _csv(cfg.get("k_values", "1,2,4,8")),
                    "--seeds",
                    str(cfg.get("seeds", 5)),
                    "--max_fpr",
                    str(cfg.get("max_fpr", 0.01)),
                    "--min_calibration_negatives",
                    str(cfg.get("min_calibration_negatives", 1000)),
                ]
                if (
                    cfg.get("overwrite", False)
                    and baseline_key not in overwritten_baselines
                ):
                    text_cmd.append("--overwrite")
                    overwritten_baselines.add(baseline_key)
                run_cmd(text_cmd)
            if "B2_text_embedding_logistic" in registered_baselines:
                baseline_key = ("B2_text_embedding_logistic", source_task, target_task)
                embedding_dirs = model_cfg["text_embedding_cache_dirs"]
                embedding_cmd = [
                    sys.executable,
                    "-m",
                    "cli.run_embedding_baselines",
                    "--source_task",
                    source_task,
                    "--source_cache_dir",
                    embedding_dirs[source_task],
                    "--target_task",
                    target_task,
                    "--target_cache_dir",
                    embedding_dirs[target_task],
                    "--calibration_task",
                    "benign_calibration",
                    "--calibration_cache_dir",
                    model_cfg["benign_embedding_cache_dir"],
                    "--model",
                    model_name,
                    "--results_dir",
                    str(results_dir),
                    "--views",
                    text_views,
                    "--k_values",
                    _csv(cfg.get("k_values", "1,2,4,8")),
                    "--seeds",
                    str(cfg.get("seeds", 5)),
                    "--max_fpr",
                    str(cfg.get("max_fpr", 0.01)),
                    "--min_calibration_negatives",
                    str(cfg.get("min_calibration_negatives", 1000)),
                ]
                if (
                    cfg.get("overwrite", False)
                    and baseline_key not in overwritten_baselines
                ):
                    embedding_cmd.append("--overwrite")
                    overwritten_baselines.add(baseline_key)
                run_cmd(embedding_cmd)
            judge_probes = {
                "B3_llm_judge_zero_shot",
                "B3_llm_judge_few_shot",
            }
            if registered_baselines.intersection(judge_probes):
                baseline_key = ("B3_llm_judge", source_task, target_task)
                judge_cmd = [
                    sys.executable,
                    "-m",
                    "cli.run_llm_judge_baselines",
                    "--source_task",
                    source_task,
                    "--source_data",
                    labeled_data[source_task],
                    "--target_task",
                    target_task,
                    "--target_data",
                    labeled_data[target_task],
                    "--calibration_task",
                    "benign_calibration",
                    "--calibration_data",
                    benign_data,
                    "--judge_config",
                    cfg["llm_judge_model_lock"],
                    "--judge_model_key",
                    cfg["llm_judge_model_key"],
                    "--judge_cache_dir",
                    model_cfg["llm_judge_cache_dir"],
                    "--judge_batch_size",
                    str(cfg.get("llm_judge_batch_size", 8)),
                    "--model",
                    model_name,
                    "--results_dir",
                    str(results_dir),
                    "--views",
                    _csv(cfg["llm_judge_views"]),
                    "--modes",
                    _csv(cfg["llm_judge_modes"]),
                    "--k_values",
                    _csv(cfg.get("k_values", "1,2,4,8")),
                    "--seeds",
                    str(cfg.get("seeds", 5)),
                    "--max_fpr",
                    str(cfg.get("max_fpr", 0.01)),
                    "--min_calibration_negatives",
                    str(cfg.get("min_calibration_negatives", 1000)),
                    "--device",
                    judge_device,
                ]
                if (
                    cfg.get("overwrite", False)
                    and baseline_key not in overwritten_baselines
                ):
                    judge_cmd.append("--overwrite")
                    overwritten_baselines.add(baseline_key)
                run_cmd(judge_cmd)
            if (
                "B4_output_confidence_logistic" in registered_baselines
                and _baseline_available(
                    source_task, "B4_output_confidence_logistic"
                )
                and _baseline_available(
                    target_task, "B4_output_confidence_logistic"
                )
            ):
                baseline_key = (
                    "B4_output_confidence_logistic",
                    source_task,
                    target_task,
                )
                confidence_cmd = [
                    sys.executable,
                    "-m",
                    "cli.run_output_confidence_baselines",
                    "--source_task",
                    source_task,
                    "--source_data",
                    labeled_data[source_task],
                    "--target_task",
                    target_task,
                    "--target_data",
                    labeled_data[target_task],
                    "--calibration_task",
                    "benign_calibration",
                    "--calibration_data",
                    benign_data,
                    "--model",
                    model_name,
                    "--results_dir",
                    str(results_dir),
                    "--k_values",
                    _csv(cfg.get("k_values", "1,2,4,8")),
                    "--seeds",
                    str(cfg.get("seeds", 5)),
                    "--max_fpr",
                    str(cfg.get("max_fpr", 0.01)),
                    "--min_calibration_negatives",
                    str(cfg.get("min_calibration_negatives", 1000)),
                ]
                if (
                    cfg.get("overwrite", False)
                    and baseline_key not in overwritten_baselines
                ):
                    confidence_cmd.append("--overwrite")
                    overwritten_baselines.add(baseline_key)
                run_cmd(confidence_cmd)
            elif "B4_output_confidence_logistic" in registered_baselines:
                unavailable = [
                    task_name
                    for task_name in (source_task, target_task)
                    if not _baseline_available(
                        task_name, "B4_output_confidence_logistic"
                    )
                ]
                print(
                    "skipped B4_output_confidence_logistic for "
                    f"{source_task} -> {target_task}: unavailable for {unavailable}"
                )

    run_cmd(
        [
            sys.executable,
            "-m",
            "cli.aggregate_task_results",
            "--results_dir",
            str(results_dir),
            "--bootstrap_samples",
            str(cfg.get("bootstrap_samples", 500)),
        ]
    )
    if cfg.get("run_falsification_suite", False):
        manifests = sorted(
            {
                str(path)
                for model_cfg in cfg["models"]
                for path in model_cfg["falsification_manifests"].values()
            }
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "cli.evaluate_falsification_slices",
                "--results_dir",
                str(results_dir),
                "--registry",
                str(cfg["falsification_registry"]),
                "--manifests",
                *manifests,
            ]
        )
        if cfg.get("falsification_comparisons_file"):
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "cli.compute_falsification_significance",
                    "--results_dir",
                    str(results_dir),
                    "--registry",
                    str(cfg["falsification_registry"]),
                    "--comparisons",
                    str(cfg["falsification_comparisons_file"]),
                    "--bootstrap_samples",
                    str(cfg.get("bootstrap_samples", 5000)),
                ]
            )
    run_cmd(
        [
            sys.executable,
            "-m",
            "cli.build_frozen_transfer_report",
            "--results_dir",
            str(results_dir),
        ]
    )
    run_cmd(
        [
            sys.executable,
            "-m",
            "cli.build_cross_model_tables",
            "--results_dir",
            str(results_dir),
        ]
    )
    run_cmd(
        [
            sys.executable,
            "-m",
            "cli.build_protocol_split_report",
            "--results_dir",
            str(results_dir),
        ]
    )
    if cfg.get("comparisons_file"):
        run_cmd(
            [
                sys.executable,
                "-m",
                "cli.compute_task_significance",
                "--results_dir",
                str(results_dir),
                "--comparisons",
                str(cfg["comparisons_file"]),
                "--bootstrap_samples",
                str(cfg.get("bootstrap_samples", 5000)),
            ]
        )
    if cfg.get("matrix_probe") and cfg.get("matrix_k") is not None:
        run_cmd(
            [
                sys.executable,
                "-m",
                "cli.build_transfer_matrix",
                "--results_dir",
                str(results_dir),
                "--probe",
                str(cfg["matrix_probe"]),
                "--k",
                str(cfg["matrix_k"]),
            ]
        )
    run_cmd(
        [
            sys.executable,
            "-m",
            "cli.plot_camera_ready_task_results",
            "--results_dir",
            str(results_dir),
        ]
    )
    run_cmd(
        [
            sys.executable,
            "-m",
            "cli.build_geometry_report",
            "--config",
            args.config,
            "--results_dir",
            str(results_dir),
        ]
    )
    if cfg.get("run_steering", False):
        steering_cmd = [
            sys.executable,
            "-m",
            "cli.run_steering_suite",
            "--config",
            args.config,
            "--results_dir",
            str(results_dir),
        ]
        if args.device is not None:
            steering_cmd.extend(["--device", args.device])
        run_cmd(
            steering_cmd
        )


if __name__ == "__main__":
    main()
