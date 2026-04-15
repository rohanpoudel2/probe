# Final Paper Runbook

This is the exact command order for a full NeurIPS-style paper run with the current repo.

Assumptions:

- You use `Qwen3-4B` and `Qwen3-8B` in the main paper.
- You use `Gemma-2-9b` as an appendix or robustness model.
- You normalize or otherwise prepare local JSONL files for:
  - `data/final/sycophancy_main.jsonl`
  - `data/final/sycophancy_are_you_sure.jsonl`
  - `data/final/sycophancy_feedback.jsonl`
  - `data/final/motivated_reasoning_main.jsonl`
  - `data/final/motivated_reasoning_appendix.jsonl`
  - `data/final/cot_distortion_main.jsonl`
  - `data/final/honesty_control_mask.jsonl`

If your sources are already local JSONL files, skip the Hugging Face normalization step and place them at the paths above.

## 0. Environment and scope lock

```bash
export UV_CACHE_DIR=.uv-cache
export WANDB_DISABLED=true
uv venv
source .venv/bin/activate
uv sync
huggingface-cli login
python scripts/phase0_audit.py --config config.yaml
```

Your Hugging Face account must already have access to the gated `MASK` dataset, or the raw fetch step will fail there.

## 1. Fetch raw locked sources

Use `experiments/data/huggingface_source_lock.yaml` as the authoritative record for exact upstream Hugging Face repos, revisions, and file-level paths. Fetch the exact raw sources with `scripts/fetch_exact_hf_sources.py`, then build the benchmark-ready `data/final/` datasets with `scripts/build_exact_paper_datasets.py`.

## 2. Build the canonical paper dataset bundle

```bash
python scripts/fetch_exact_hf_sources.py --source all --output_dir data/raw_sources
python scripts/build_exact_paper_datasets.py --raw_dir data/raw_sources --output_dir data/final
```

## 3. Extract activations for the main paper models

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

## 4. Validate the configs

```bash
python validate_multimodel_config.py --config experiments/protocol/neurips_main_manifest.yaml --check_paths
python validate_multimodel_config.py --config experiments/multimodel/real_models_config.yaml --check_paths
python validate_multimodel_config.py --config experiments/controls/neurips_final_manifest.yaml --check_paths
python validate_multimodel_config.py --config experiments/controls/honesty_auxiliary_manifest.yaml --check_paths
```

## 5. Run the main NeurIPS benchmark

```bash
python run_multimodel_task_benchmark.py --config experiments/multimodel/real_models_config.yaml
python run_paper_benchmark.py --config experiments/protocol/neurips_main_manifest.yaml
```

## 6. Run the main ablations

```bash
python run_ablation_suite.py --config experiments/protocol/ablation_suite.yaml
```

## 7. Run the MASK auxiliary honesty-control benchmark

```bash
python run_paper_benchmark.py --config experiments/controls/honesty_auxiliary_manifest.yaml
```

## 8. Run within-family sycophancy transfer

```bash
python run_task_sweep.py --source_dir outputs/final_features/Qwen3-4B/sycophancy_main --source_task sycophancy --target_dir outputs/final_features/Qwen3-4B/sycophancy_are_you_sure --target_task sycophancy --model Qwen3-4B --results_dir results/within_family_transfer --views answer --layers all --probes P1_logistic,P2_mass_mean,P3_lda,P7_mahalanobis --k_values 1,4,8 --seeds 10 --balance_modes balanced --overwrite
python run_task_sweep.py --source_dir outputs/final_features/Qwen3-8B/sycophancy_main --source_task sycophancy --target_dir outputs/final_features/Qwen3-8B/sycophancy_are_you_sure --target_task sycophancy --model Qwen3-8B --results_dir results/within_family_transfer --views answer --layers all --probes P1_logistic,P2_mass_mean,P3_lda,P7_mahalanobis --k_values 1,4,8 --seeds 10 --balance_modes balanced
python -m cli.aggregate_task_results --results_dir results/within_family_transfer --bootstrap_samples 1000
python -m cli.compute_task_significance --results_dir results/within_family_transfer
python -m cli.build_transfer_matrix --results_dir results/within_family_transfer
```

## 9. Run the appendix and release path

```bash
python run_release_pipeline.py --base_config experiments/controls/neurips_final_manifest.yaml --controls_config experiments/controls/negative_controls.yaml
python run_ablation_suite.py --config experiments/controls/appendix_ablation_suite.yaml
```

## 10. Build paper artifacts and camera-ready outputs

```bash
python build_numbered_draft_assets.py --results_dir phase5_neurips_main_results --mapping experiments/submission/draft_asset_mapping.yaml
python generate_result_narratives.py --results_dir phase5_neurips_main_results
python package_camera_ready_bundle.py --results_dir phase5_neurips_main_results --mapping experiments/submission/draft_asset_mapping.yaml
python build_submission_manifest.py --bundle_dir phase5_neurips_main_results/camera_ready_bundle
```

## 11. Useful spot checks

```bash
python scripts/build_exact_paper_datasets.py --raw_dir data/raw_sources --output_dir data/final
```
