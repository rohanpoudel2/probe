# Few-Shot Internal Monitoring for Socially Misaligned Reasoning

This repo is the cleaned NeurIPS main-track pipeline for:

- `sycophancy` as the anchor task
- `motivated_reasoning` as the transfer task
- `cot_distortion` as the stress test
- `honesty_control` as a separate auxiliary MASK analysis

The main paper scope is intentionally narrow:

- models: `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`
- appendix model: `google/gemma-2-9b`
- core probes: `P1_logistic`, `P2_mass_mean`, `P3_lda`, `P7_mahalanobis`
- main `k` values: `1,4,8`
- main training mode: `balanced`

## Maintained Files

These are the configs and scripts that define the real paper path:

- `experiments/data/huggingface_source_lock.yaml`
- `experiments/protocol/neurips_main_manifest.yaml`
- `experiments/multimodel/real_models_config.yaml`
- `experiments/protocol/ablation_suite.yaml`
- `experiments/controls/honesty_auxiliary_manifest.yaml`
- `experiments/controls/neurips_final_manifest.yaml`
- `experiments/controls/appendix_ablation_suite.yaml`
- `experiments/controls/negative_controls.yaml`
- `experiments/submission/draft_asset_mapping.yaml`
- `scripts/fetch_exact_hf_sources.py`
- `scripts/build_exact_paper_datasets.py`

## Environment

```bash
export UV_CACHE_DIR=.uv-cache
export WANDB_DISABLED=true
uv venv
source .venv/bin/activate
uv sync
huggingface-cli login
python scripts/phase0_audit.py --config config.yaml
```

`UV_CACHE_DIR=.uv-cache` keeps `uv` cache writes inside the repo instead of depending on a machine-global cache path. `WANDB_DISABLED=true` prevents unnecessary `wandb` side effects when `sae-lens` is imported. Run `huggingface-cli login` once before the dataset fetch stage. This is required for the gated `MASK` dataset and also helps avoid model-download auth issues. Your Hugging Face account still needs to have `MASK` access approved, or the fetch stage will fail on that dataset.

## Exact Upstream Dataset Lock

The exact upstream raw sources are locked in `experiments/data/huggingface_source_lock.yaml`.

Locked sources:

- `SycophancyEval`: `meg-tong/sycophancy-eval`
- `MMLU`: `cais/mmlu`
- `ARC-Challenge`: `allenai/ai2_arc`
- `CommonsenseQA`: `tau/commonsense_qa`
- `AQuA-RAT`: `deepmind/aqua_rat`
- `MASK`: `cais/MASK`
- `MonitorBench`: GitHub-only lock for now, no verified official HF dataset path in this repo

Important:

- the repo now automates benchmark construction from these raw sources with `scripts/build_exact_paper_datasets.py`
- the builders use deterministic templated conversions for `SycophancyEval`, motivated-reasoning source banks, `MASK`, and `MonitorBench`
- `MonitorBench` is fetched from GitHub because no verified official HF dataset path is pinned here

## Dataset Commands

### 1. Fetch the exact raw locked sources

```bash
python scripts/fetch_exact_hf_sources.py --source all --output_dir data/raw_sources
```

These commands write raw source exports under `data/raw_sources/`.

### 2. Build the benchmark-ready paper datasets automatically

```bash
python scripts/build_exact_paper_datasets.py --raw_dir data/raw_sources --output_dir data/final
```

After this step, the paper pipeline expects:

- `data/final/sycophancy_main.jsonl`
- `data/final/sycophancy_are_you_sure.jsonl`
- `data/final/sycophancy_feedback.jsonl`
- `data/final/motivated_reasoning_main.jsonl`
- `data/final/motivated_reasoning_appendix.jsonl`
- `data/final/cot_distortion_main.jsonl`
- `data/final/honesty_control_mask.jsonl`

This is the maintained no-manual dataset path.

## Activation Extraction

### Qwen3-4B

```bash
python extract_task_activations.py --task sycophancy --data data/final/sycophancy_main.jsonl --model Qwen/Qwen3-4B --layers 8,16,24,32 --views full_text,pressure_context,answer --output_dir outputs/final_features/Qwen3-4B/sycophancy_main --modified_modes standard,prompted
python extract_task_activations.py --task motivated_reasoning --data data/final/motivated_reasoning_main.jsonl --model Qwen/Qwen3-4B --layers 8,16,24,32 --views full_text,hint_context,reasoning,reasoning_early,reasoning_mid,reasoning_late,answer --output_dir outputs/final_features/Qwen3-4B/motivated_reasoning_main --modified_modes standard,prompted
python extract_task_activations.py --task cot_distortion --data data/final/cot_distortion_main.jsonl --model Qwen/Qwen3-4B --layers 8,16,24,32 --views full_text,reasoning,reasoning_early,reasoning_mid,reasoning_late,pre_answer,answer --output_dir outputs/final_features/Qwen3-4B/cot_distortion_main --modified_modes standard,prompted
python extract_task_activations.py --task sycophancy --data data/final/sycophancy_are_you_sure.jsonl --model Qwen/Qwen3-4B --layers 8,16,24,32 --views full_text,pressure_context,answer --output_dir outputs/final_features/Qwen3-4B/sycophancy_are_you_sure --modified_modes standard,prompted
python extract_task_activations.py --task sycophancy --data data/final/sycophancy_feedback.jsonl --model Qwen/Qwen3-4B --layers 8,16,24,32 --views full_text,pressure_context,answer --output_dir outputs/final_features/Qwen3-4B/sycophancy_feedback --modified_modes standard,prompted
python extract_task_activations.py --task honesty_control --data data/final/honesty_control_mask.jsonl --model Qwen/Qwen3-4B --layers 8,16,24,32 --views full_text,reasoning,answer --output_dir outputs/final_features/Qwen3-4B/honesty_control_mask --modified_modes standard,prompted
```

### Qwen3-8B

```bash
python extract_task_activations.py --task sycophancy --data data/final/sycophancy_main.jsonl --model Qwen/Qwen3-8B --layers 8,16,24,32 --views full_text,pressure_context,answer --output_dir outputs/final_features/Qwen3-8B/sycophancy_main --modified_modes standard,prompted
python extract_task_activations.py --task motivated_reasoning --data data/final/motivated_reasoning_main.jsonl --model Qwen/Qwen3-8B --layers 8,16,24,32 --views full_text,hint_context,reasoning,reasoning_early,reasoning_mid,reasoning_late,answer --output_dir outputs/final_features/Qwen3-8B/motivated_reasoning_main --modified_modes standard,prompted
python extract_task_activations.py --task cot_distortion --data data/final/cot_distortion_main.jsonl --model Qwen/Qwen3-8B --layers 8,16,24,32 --views full_text,reasoning,reasoning_early,reasoning_mid,reasoning_late,pre_answer,answer --output_dir outputs/final_features/Qwen3-8B/cot_distortion_main --modified_modes standard,prompted
python extract_task_activations.py --task sycophancy --data data/final/sycophancy_are_you_sure.jsonl --model Qwen/Qwen3-8B --layers 8,16,24,32 --views full_text,pressure_context,answer --output_dir outputs/final_features/Qwen3-8B/sycophancy_are_you_sure --modified_modes standard,prompted
python extract_task_activations.py --task sycophancy --data data/final/sycophancy_feedback.jsonl --model Qwen/Qwen3-8B --layers 8,16,24,32 --views full_text,pressure_context,answer --output_dir outputs/final_features/Qwen3-8B/sycophancy_feedback --modified_modes standard,prompted
python extract_task_activations.py --task honesty_control --data data/final/honesty_control_mask.jsonl --model Qwen/Qwen3-8B --layers 8,16,24,32 --views full_text,reasoning,answer --output_dir outputs/final_features/Qwen3-8B/honesty_control_mask --modified_modes standard,prompted
```

### Gemma-2-9b appendix model

```bash
python extract_task_activations.py --task sycophancy --data data/final/sycophancy_main.jsonl --model google/gemma-2-9b --layers 10,20,30,42 --views full_text,pressure_context,answer --output_dir outputs/final_features/Gemma-2-9b/sycophancy_main --modified_modes standard,prompted
python extract_task_activations.py --task motivated_reasoning --data data/final/motivated_reasoning_main.jsonl --model google/gemma-2-9b --layers 10,20,30,42 --views full_text,hint_context,reasoning,reasoning_early,reasoning_mid,reasoning_late,answer --output_dir outputs/final_features/Gemma-2-9b/motivated_reasoning_main --modified_modes standard,prompted
python extract_task_activations.py --task cot_distortion --data data/final/cot_distortion_main.jsonl --model google/gemma-2-9b --layers 10,20,30,42 --views full_text,reasoning,reasoning_early,reasoning_mid,reasoning_late,pre_answer,answer --output_dir outputs/final_features/Gemma-2-9b/cot_distortion_main --modified_modes standard,prompted
python extract_task_activations.py --task honesty_control --data data/final/honesty_control_mask.jsonl --model google/gemma-2-9b --layers 10,20,30,42 --views full_text,reasoning,answer --output_dir outputs/final_features/Gemma-2-9b/honesty_control_mask --modified_modes standard,prompted
```

## Main-Track Run Commands

### Validate configs

```bash
python validate_multimodel_config.py --config experiments/protocol/neurips_main_manifest.yaml --check_paths
python validate_multimodel_config.py --config experiments/multimodel/real_models_config.yaml --check_paths
python validate_multimodel_config.py --config experiments/controls/neurips_final_manifest.yaml --check_paths
python validate_multimodel_config.py --config experiments/controls/honesty_auxiliary_manifest.yaml --check_paths
```

### Main benchmark

```bash
python run_multimodel_task_benchmark.py --config experiments/multimodel/real_models_config.yaml
python run_paper_benchmark.py --config experiments/protocol/neurips_main_manifest.yaml
```

### Main ablations

```bash
python run_ablation_suite.py --config experiments/protocol/ablation_suite.yaml
```

### Auxiliary MASK benchmark

```bash
python run_paper_benchmark.py --config experiments/controls/honesty_auxiliary_manifest.yaml
```

### Within-family sycophancy transfer

```bash
python run_task_sweep.py --source_dir outputs/final_features/Qwen3-4B/sycophancy_main --source_task sycophancy --target_dir outputs/final_features/Qwen3-4B/sycophancy_are_you_sure --target_task sycophancy --model Qwen3-4B --results_dir results/within_family_transfer --views answer --layers all --probes P1_logistic,P2_mass_mean,P3_lda,P7_mahalanobis --k_values 1,4,8 --seeds 10 --balance_modes balanced --overwrite
python run_task_sweep.py --source_dir outputs/final_features/Qwen3-8B/sycophancy_main --source_task sycophancy --target_dir outputs/final_features/Qwen3-8B/sycophancy_are_you_sure --target_task sycophancy --model Qwen3-8B --results_dir results/within_family_transfer --views answer --layers all --probes P1_logistic,P2_mass_mean,P3_lda,P7_mahalanobis --k_values 1,4,8 --seeds 10 --balance_modes balanced
python -m cli.aggregate_task_results --results_dir results/within_family_transfer --bootstrap_samples 1000
python -m cli.compute_task_significance --results_dir results/within_family_transfer
python -m cli.build_transfer_matrix --results_dir results/within_family_transfer
```

### Appendix and release

```bash
python run_release_pipeline.py --base_config experiments/controls/neurips_final_manifest.yaml --controls_config experiments/controls/negative_controls.yaml
python run_ablation_suite.py --config experiments/controls/appendix_ablation_suite.yaml
```

### Camera-ready artifacts

```bash
python build_numbered_draft_assets.py --results_dir phase5_neurips_main_results --mapping experiments/submission/draft_asset_mapping.yaml
python generate_result_narratives.py --results_dir phase5_neurips_main_results
python package_camera_ready_bundle.py --results_dir phase5_neurips_main_results --mapping experiments/submission/draft_asset_mapping.yaml
python build_submission_manifest.py --bundle_dir phase5_neurips_main_results/camera_ready_bundle
```

## Operational Notes

- Run `phase5`, `phase7`, and `phase8` style pipelines sequentially, not in parallel. They reuse shared temp/result locations.
- `MASK` stays auxiliary. Do not fold it into the main sycophancy benchmark.
- The repo no longer treats sample or smoke configs as the primary path.

## Paper Docs

- [docs/neurips_main_track_map.md](docs/neurips_main_track_map.md)
- [docs/final_paper_runbook.md](docs/final_paper_runbook.md)
- [docs/final_paper_experiment_commands.tex](docs/final_paper_experiment_commands.tex)
