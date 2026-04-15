# NeurIPS Main-Track Map

This repo can still run a broad benchmark, but the main-track paper path should stay narrower than the full code surface.

## Core paper scope

- Tasks: keep the existing three-task story, with `sycophancy` as the primary source task and `motivated_reasoning` plus `cot_distortion` as transfer and stress-test targets.
- Auxiliary control: use `honesty_control` for MASK-style honesty-vs-accuracy analysis outside the main benchmark claim.
- Methods: center the main text on `P1_logistic`, `P2_mass_mean`, `P3_lda`, and `P7_mahalanobis`.
- Models: use two models in the main paper. The current main-track templates keep `Qwen3-4B` and `Qwen3-8B`. Use `Gemma-2-9b` as a cross-family appendix model.
- Training regime: use `balanced` few-shot training in the main text. Treat `imbalanced` as an appendix robustness check.
- Few-shot settings: keep `k in {1, 4, 8}` in the main paper.
- Views: make `answer` and `reasoning` the main views. These are the most comparable across all three task families and best match the internal-monitoring story.

## Keep In Main Text

- Same-task calibration on `sycophancy`.
- Cross-task transfer from `sycophancy` to `motivated_reasoning`.
- Cross-task transfer from `sycophancy` to `cot_distortion`.
- Layerwise analysis if it helps explain where the effect emerges.
- Geometry and steering only as supporting evidence for the main claim, not as independent benchmark tracks.

## Move To Appendix

- Extra probe families such as `P4_cosine`, `P5_sae`, and `P6_prompted`.
- Extra models beyond the main two.
- Extra views such as `full_text`, `pressure_context`, `hint_context`, and `pre_answer` unless they directly answer a main-text question.
- Extra training regimes and wide method sweeps.
- Release-oriented controls and stress tests that are useful for completeness but not central to the main claim.
- `MASK` and other honesty-vs-accuracy checks should be reported as auxiliary controls rather than folded into the main sycophancy path.

## File Mapping

- `experiments/protocol/neurips_main_manifest.yaml`: trimmed main-text protocol config.
- `experiments/multimodel/real_models_config.yaml`: trimmed multi-model transfer config.
- `experiments/protocol/ablation_suite.yaml`: targeted ablations for view dependence, low-shot sensitivity, and training-regime robustness.
- `experiments/controls/neurips_final_manifest.yaml`: broader appendix and release config.
- `experiments/controls/appendix_ablation_suite.yaml`: wider appendix ablations.
- `experiments/controls/honesty_auxiliary_manifest.yaml`: separate MASK-style auxiliary control config.

## Suggested Paper Shape

1. Problem setup: few-shot monitoring of socially misaligned reasoning.
2. Core empirical result: simple linear monitors work well in the low-shot regime.
3. Transfer result: detectors learned on sycophancy transfer to motivated reasoning and CoT distortion.
4. Diagnostic result: answer and reasoning views, plus layer structure, explain where the signal lives.
5. Mechanistic support: geometry and steering show that the learned directions are behaviorally meaningful.
6. Appendix: extra models, extra methods, imbalanced training, and broader controls.
