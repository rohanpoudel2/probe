# Implementation Status

## Implemented and tested

- Prompt-only scenario schema with content hashes and immutable source provenance.
- On-policy, resumable rollout generation with pinned model/tokenizer revisions.
- CPU, CUDA, and Apple-MPS device resolution across model-backed commands.
- A pinned MonitorBench importer for the test-only `cot_distortion` family. Its
  upstream evaluated artifacts use MonitorBench's own runtime; Apple-MPS is
  supported from verified import through every downstream stage.
- Generation-time confidence traces bound to sampled token IDs.
- Post-generation executable behavior labeling with retained non-target errors and multidimensional outcome metadata.
- Immutable group-aware behavior `train`/`eval`/`test` partitions and matched few-shot sampling.
- Unlabeled, on-policy reference traffic with deterministic calibration/test partitions.
- Reference membership validation that explicitly forbids semantic-negative claims.
- Split-conformal upper-tail calibration at a 1% reference alert budget.
- Disjoint reference holdout scoring with alert counts and 95% Wilson intervals.
- Claim-status reporting that distinguishes supported, inconclusive, and violated operational budgets from a conservative seedwise Wilson-interval envelope.
- Activation probes: logistic, mass mean, LDA, cosine direction, shrinkage Mahalanobis, and the lightweight sequence-aware P8 trajectory monitor.
- Assistant-response-relative prefix extraction, prompt-end controls, matched visible-text prefix baselines, frozen prefix-system selection, and hierarchical AUEW inference.
- Matched black-box systems: TF-IDF, frozen text embeddings, independent zero/few-shot LLM judge, and generation-confidence logistic regression. Zero-shot judge rows are replicated across registered label budgets without consuming labels.
- Non-confirmatory attack utilities for output dilution, reasoning compression, monitor-aware prompting, and surrogate-loss construction.
- Non-confirmatory causal utilities for activation patching, steering, nuisance projection, and norm-matched controls.
- Source-only system selection with deterministic tie-breaking.
- Selection-evidence verification and immutable generation of confirmatory manifests and exact comparison registries.
- Content-hashed text-embedding and LLM-judge caches.
- Registry-bound behavior, domain, template, presentation, and obfuscation shifts.
- Executable verbatim-wrapper and reversible-ROT13 shift generation.
- Exact-prompt hard-negative pairing and pairwise score analysis.
- Paired hierarchical seed/group inference with finite-bootstrap correction and registered multiplicity control.
- Multi-model protocol validation requiring three distinct monitored-model families.
- Dry-run orchestration, resumable result writing, and result-artifact gates.
- Unit and integration coverage for the maintained execution paths.

## Required before confirmatory execution

- Generate all model-specific behavior and reference rollouts.
- Regenerate any reference response that reaches the output-length cap.
- Produce objective labels and verify both behavior classes for every maintained task.
- Produce 10,000 independent reference-calibration and 10,000 independent reference-holdout groups per monitored model revision.
- Extract every registered activation layer and text-embedding view without truncation.
- Build at least 100 independent held-out groups for every enabled shift axis.
- Build at least 100 exact-prompt hard-negative pairs for every enabled task.
- Run the source-only selection manifest and retain its complete run summaries and `early_warning_source_selection.csv`.
- Run `freeze_protocol_from_selection`; it re-derives both static and early-warning selectors, verifies that no behavior-test metric was scored, and embeds the exact prefix identities in immutable comparison files plus a frozen confirmatory manifest.
- Run `validate_multimodel_config --check_paths --final_protocol` on the generated manifest.
- Execute the complete model/task/seed matrix without partial-run suppression.

## Interpretation

The implementation is ready to generate and evaluate the required evidence, but the research is not complete until the registered experiments have run. A null white-box gain, excessive held-out reference alert rate, weak transfer, or hard-negative failure remains a valid outcome.
