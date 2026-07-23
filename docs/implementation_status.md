# Implementation Status

## Implemented and tested

- Prompt-only, content-addressed scenario schema with immutable protocol splits.
- Prompt-only adapters for pinned SycophancyEval and motivated-reasoning MCQ sources.
- Resumable on-policy Hugging Face rollout generation with chat-template/revision provenance and hash-bound per-token confidence traces.
- Scenario schema v2 with hash-bound behavior/domain/template/paraphrase/obfuscation assignments and group-level held-out isolation.
- Independently reviewed paraphrase/obfuscation merge with exact parentage and prompt hashes.
- Conservative executable answer-labeling rule with explicit ambiguous-example exclusions.
- Separate annotation merge and rollout-integrity audit.
- Chat-faithful activation extraction with zero-based transformer-block semantics.
- Strict span/truncation alignment and feature provenance hashes.
- Matched scenario-group few-shot sampling.
- Separate calibration split and frozen-threshold evaluation.
- Atomic per-run, per-example prediction artifacts.
- Prompt-, answer-, and transcript-only TF-IDF leakage baselines.
- Pinned Qwen3 transformer-embedding caches with documented last-token pooling, L2 normalization, truncation gates, and matched few-shot logistic sweeps.
- Pinned independent Phi-4 zero/few-shot judge with contextual forced-choice token validation, no-truncation gates, and resumable content-hashed score caches.
- Matched output-confidence logistic baseline over a frozen 22-feature generation-trace representation.
- Pinned WildChat natural-traffic candidate sampling that never copies upstream assistant outputs.
- Model-identity-blinded two-rater benign screening with exact-text hashes, conservative exclusions, and agreement reporting.
- A calibration-only task loader plus merge/audit gates that reject silently assigned benign labels.
- Release-orchestrator and final-validator wiring for TF-IDF, cached embeddings, zero/few-shot judging, and output confidence.
- Exact-trigger-prompt hard-negative manifests plus per-monitor row-level shift and paired-hard-negative evaluation artifacts.
- Content-hashed falsification comparison registration with exact system selectors, hierarchical paired inference, global Holm correction, and immutable artifact archiving.
- Standardized logistic and simple direction probes with minimum-sample guards.
- Hierarchical paired inference over training seeds and scenario groups.
- Holm correction for pre-registered comparison families.
- Non-oracle fixed-`k` layer/view selection and fixed-system transfer matrices.
- Meaningful train-only negative controls.
- Pinned dataset/archive sources, `uv.lock`, CI, and integrity tests.
- Pinned official MonitorBench adapter covering all 19 tasks and 69 valid task/stress artifacts, with safe revision-specific extraction, archived-source/tree manifests, immutable run provenance, strict tested-artifact parsing, and test-only verified-outcome normalization.
- Exact-prompt MonitorBench hard-negative pairing from repeated official verifier outcomes, with explicit separation from the upstream monitorability metric and a frozen B4 inapplicability instead of reconstructed confidence.
- Final-execution validation gates and fail-loud artifact packaging.
- MASK auxiliary now uses on-policy neutral-belief versus pressured-statement scenarios with registered task-label semantics.

## Required before the leakage pilot

- Finish and validate the exact scenario inventory intended for the pilot.
- Commit the protocol so generation and extraction have a clean code revision.
- Select one pilot model revision and a manageable fixed layer set.
- Generate and independently screen at least 1,000 on-policy benign responses for the pilot.
- Freeze executable labeling rules and human-annotation rubrics; audit a blinded sample to estimate label error.
- Materialize the registered text-embedding caches for every pilot task/view and monitored model.
- Materialize the registered LLM-judge score caches from a clean code revision.
- Generate stochastic repeated behavior rollouts and verify that the pilot yields registered exact-prompt hard-negative pairs.
- Materialize and independently review the pilot paraphrase and obfuscation prompt variants.

## Required before protocol freeze
- Materialize the complete official MonitorBench task/stress matrix for every monitored model, preserve upstream monitor outputs and metrics separately, and pass the strict importer without `--allow_partial_pilot`.
- Materialize at least the power-analysis-backed number of independently reviewed held-out groups for every shift axis.
- Materialize at least the power-analysis-backed number of exact-prompt hard-negative groups for every enabled task.
- Build additional natural benign traffic from independent sources.
- Choose at least three genuinely different current model families and pin commits.
- Conduct power and tail-precision analysis for the final benign sample size.
- Copy and complete both pre-registered comparison templates; register exact primary/falsification systems, label budgets, exclusions, five shift axes, enabled hard-negative tasks, and the global comparison families before any final-test prediction exists.

## Required before the full run

- Use at least 10,000 independent benign calibration groups; repeated rollouts do not count as new groups.
- Validate all feature directories with `--final_protocol --check_paths`.
- Confirm the final-test prediction files do not exist before registration is signed.
- Run a clean end-to-end smoke fixture through scenario, rollout, annotation, extraction, baselines, probe sweep, and inference.
- Archive code, configs, source hashes, annotations, predictions, and adjudication records.

## Deliberately deferred

- Generation-time temporal monitoring before the visible answer.
- Behavioral activation steering, SAE comparisons, and prompted-probe comparisons.

These are follow-up contributions, not prerequisites for deciding whether few-shot white-box monitoring adds value over visible transcripts.
