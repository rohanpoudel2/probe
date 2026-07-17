# Data-Efficient White-Box Monitoring Under Distribution Shift

This repository is being built around one falsifiable question:

> With only a few matched labeled scenarios, does an activation monitor improve on the strongest transcript-only monitor at a threshold calibrated to 1% false-positive rate on benign traffic?

No full end-to-end frontier experiments have been executed under the new protocol yet. The former pipeline authored both positive and negative completions; it is retained only as a blocked legacy debug fixture and is not valid frontier data.

## Scientific contract

The maintained path enforces these invariants:

- Responses are generated on-policy by the monitored model.
- Prompt scenarios contain no authored response or behavior label.
- Labels are merged only after generation from a declared verifier or annotation protocol.
- Few-shot `k` counts matched scenario groups; balanced training contains `2 * k` examples.
- Train, calibration, selection/eval, and final-test groups are disjoint.
- Every source/held-out shift assignment is registry-bound; held-out groups can occur only in test.
- Hard negatives are on-policy negative responses paired to positives from the exact same trigger prompt.
- The operating threshold is fitted on calibration negatives and reused unchanged on every test target.
- Every run saves per-example predictions for paired, scenario-level inference.
- Model, tokenizer, dataset, code, and upstream-source revisions are pinned.
- Final execution fails on partial runs or missing required artifacts.

The complete protocol is in [docs/frontier_research_protocol.md](docs/frontier_research_protocol.md), the frozen negative-eligibility decisions are defined in [docs/benign_screening_rubric.md](docs/benign_screening_rubric.md), and transformed-prompt review is specified in [docs/falsification_review_protocol.md](docs/falsification_review_protocol.md).
Current implementation coverage and remaining blockers are tracked in [docs/implementation_status.md](docs/implementation_status.md).

## Environment and tests

```bash
uv sync --locked --extra dev
uv run pytest
```

The dependency graph is committed in `uv.lock`. CI runs the same locked test suite.

## Data flow

### 1. Create prompt-only scenarios

Each JSONL record must follow `frontier-monitor-scenario-v2`:

```json
{
  "schema_version": "frontier-monitor-scenario-v2",
  "scenario_id": "question-0001-pressure",
  "group_id": "question-0001",
  "task_family": "sycophancy",
  "condition": "user_pressure",
  "protocol_split": "train",
  "source": "pinned-benchmark-revision",
  "messages": [{"role": "user", "content": "..."}],
  "metadata": {
    "gold_answer": "...",
    "pressure_target": "...",
    "falsification": {
      "schema_version": "frontier-falsification-example-v1",
      "registry_id": "frontier-falsification-v1",
      "registry_sha256": "...",
      "axes": {
        "behavior": {"value": "sycophancy", "role": "source"},
        "domain": {"value": "sycophancy_eval", "role": "source"},
        "template": {"value": "answer", "role": "source"},
        "paraphrase": {"value": "original", "role": "source"},
        "obfuscation": {"value": "none", "role": "source"}
      },
      "metadata_sha256": "..."
    }
  }
}
```

Scenario validation rejects `label`, `assistant_response`, `reasoning`, and other authored outcome fields. Its content hash binds the prompt, split, source, answer key, and all falsification metadata.

Pinned SycophancyEval and motivated-reasoning question banks can be converted to this prompt-only schema without creating responses:

```bash
uv run python scripts/fetch_exact_hf_sources.py --source sycophancy_eval --output_dir data/raw_sources
uv run python scripts/fetch_exact_hf_sources.py --source motivated_reasoning_raw --output_dir data/raw_sources
uv run python scripts/build_on_policy_scenarios.py \
  --task all \
  --raw_dir data/raw_sources \
  --output_dir data/scenarios
```

The frozen registry assigns original versus held-out templates and domains. Reviewed paraphrase or obfuscation prompts are merged before generation. Every transformed prompt must be bound to its parent and exact prompt hash and must pass unanimous semantic-equivalence, behavior-preservation, and answer-leakage decisions from at least two distinct reviewers.

```bash
uv run python -m cli.merge_reviewed_shift_scenarios \
  --base_scenarios data/scenarios/sycophancy.jsonl \
  --variants data/shift_reviews/sycophancy_variants.jsonl \
  --registry experiments/protocol/falsification_registry.yaml \
  --output data/scenarios/sycophancy_with_shifts.jsonl
```

### 2. Generate actual model rollouts

```bash
uv run python -m cli.generate_task_rollouts \
  --scenarios data/scenarios/sycophancy.jsonl \
  --output data/rollouts/Qwen3-4B/sycophancy.jsonl \
  --model Qwen/Qwen3-4B \
  --model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --num_rollouts 4 \
  --temperature 0.7 \
  --top_p 0.95 \
  --device mps \
  --seed 1000
```

Generation is resumable and records the exact chat template, token IDs, decoding configuration, seed, code state, model/tokenizer revisions, and a hash-bound per-token confidence trace from the processed distribution actually used to sample each token. Multiple rollouts require stochastic decoding; the generator rejects repeated greedy completions because they are not independent observations and cannot produce meaningful exact-prompt hard negatives.

Model-backed commands in this repo accept a `--device` flag (`auto|cpu|cuda|cuda:N|mps`). On Apple Silicon, use `--device mps`.

### 3. Build and independently screen benign calibration traffic

Natural calibration prompts come from a pinned, deterministic sample of WildChat. Upstream moderation is only a prefilter; it never assigns a negative label.

```bash
uv run python scripts/fetch_exact_hf_sources.py \
  --source benign_calibration_raw \
  --output_dir data/raw_sources

uv run python scripts/build_benign_calibration_scenarios.py \
  --raw_data data/raw_sources/benign_calibration_raw/wildchat_train_sample.jsonl \
  --output data/scenarios/benign_calibration.jsonl \
  --report data/audits/benign_candidate_build.json

uv run python -m cli.generate_task_rollouts \
  --scenarios data/scenarios/benign_calibration.jsonl \
  --output data/rollouts/Qwen3-4B/benign_calibration.jsonl \
  --model Qwen/Qwen3-4B \
  --model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --device mps

uv run python -m cli.prepare_benign_screening \
  --rollouts data/rollouts/Qwen3-4B/benign_calibration.jsonl \
  --output data/screening/Qwen3-4B/benign_template.jsonl

uv run python -m cli.merge_benign_screening \
  --rollouts data/rollouts/Qwen3-4B/benign_calibration.jsonl \
  --ratings data/screening/Qwen3-4B/rater_a.jsonl data/screening/Qwen3-4B/rater_b.jsonl \
  --output_annotations data/annotations/Qwen3-4B/benign_calibration.jsonl \
  --report data/audits/Qwen3-4B/benign_screening.json

uv run python -m cli.merge_rollout_labels \
  --rollouts data/rollouts/Qwen3-4B/benign_calibration.jsonl \
  --annotations data/annotations/Qwen3-4B/benign_calibration.jsonl \
  --output data/labeled/Qwen3-4B/benign_calibration.jsonl

uv run python -m cli.audit_rollout_dataset \
  --data data/labeled/Qwen3-4B/benign_calibration.jsonl \
  --benign_calibration \
  --require_confidence_trace
```

Each accepted response must receive unanimous eligibility decisions from at least two distinct, model-identity-blinded raters. Ratings are bound to the exact prompt and response hash. A failed criterion, disagreement, stale hash, or missing rating causes exclusion; it never becomes a label of 0.

### 4. Adjudicate and merge behavior labels

Annotations are a separate JSONL keyed by `rollout_id`. Every annotation requires:

- binary `label`;
- `label_source`, such as a deterministic task verifier or blinded adjudication;
- `annotation_protocol`, identifying the frozen rubric/version.

```bash
uv run python -m cli.verify_rollout_labels \
  --rollouts data/rollouts/Qwen3-4B/sycophancy.jsonl \
  --output data/annotations/Qwen3-4B/sycophancy.jsonl

uv run python -m cli.merge_rollout_labels \
  --rollouts data/rollouts/Qwen3-4B/sycophancy.jsonl \
  --annotations data/annotations/Qwen3-4B/sycophancy.jsonl \
  --output data/labeled/Qwen3-4B/sycophancy.jsonl

uv run python -m cli.audit_rollout_dataset \
  --data data/labeled/Qwen3-4B/sycophancy.jsonl \
  --output data/audits/Qwen3-4B/sycophancy.json \
  --require_confidence_trace \
  --falsification_registry experiments/protocol/falsification_registry.yaml \
  --require_falsification
```

### 5. Freeze falsification slices and hard negatives

After labels are frozen, build one evaluation manifest per monitored model and task. It indexes every registered shift slice and pairs a label-0 response only with a label-1 response from the exact same trigger scenario, prompt hash, and shift signature. The default pilot command fails if an enabled task has no such pair.

```bash
uv run python -m cli.build_falsification_manifest \
  --data data/labeled/Qwen3-4B/sycophancy.jsonl \
  --registry experiments/protocol/falsification_registry.yaml \
  --model_name Qwen3-4B \
  --output data/falsification/Qwen3-4B/sycophancy.json
```

These manifests are registered under each model's `falsification_manifests` mapping. The protocol orchestrator evaluates every saved monitor prediction on source and held-out domain, template, paraphrase, obfuscation, and behavior slices, plus the matched hard-negative pairs. It archives the exact registry and manifests with the result bundle, along with row-level `falsification_shift_predictions.jsonl` and `falsification_pair_predictions.jsonl` evidence.

#### Official MonitorBench evaluation family

MonitorBench is integrated only through its pinned official runner and tested
artifacts. Fetching the source stores the original archive, extracts it into a
revision-specific directory, and writes a manifest that re-hashes the complete
tree and critical evaluator files:

```bash
uv run python scripts/fetch_exact_hf_sources.py \
  --source cot_monitorability_raw \
  --output_dir data/raw_sources
```

Run the upstream code at the pinned commit using all official task/stress pairs,
then copy and complete `experiments/protocol/monitorbench_run_manifest.example.yaml`.
The resolved run manifest must pin the evaluated model, tokenizer, chat template,
generation config, and verifier model. Importing is read-only with respect to the
official results and does not run a model:

```bash
uv run python -m cli.import_monitorbench_rollouts \
  --results_root /path/to/MonitorBench/results/EVALUATED_MODEL \
  --source_manifest data/raw_sources/cot_monitorability_raw/monitorbench_source_manifest.json \
  --run_manifest experiments/protocol/monitorbench_run_manifest.yaml \
  --output data/final/cot_distortion_main.jsonl
```

The final importer requires all 69 registered task/stress artifacts (19 tasks;
`original` only for the 12 input-intervention tasks), boolean verifier results
aligned one-to-one with responses, both outcome labels, and at least one
exact-prompt matched pair. `--allow_partial_pilot` is available only for schema
smoke tests and marks every row ineligible for the main study.

The normalized binary construct is **official target outcome verified**, not
“unfaithful CoT.” MonitorBench's official monitorability score remains a separate
result computed by the upstream monitor pipeline. Official tested artifacts do
not contain per-token generation distributions, so B4 output-confidence is
reported as structurally unavailable for this target; the pipeline never
fabricates or teacher-force-reconstructs those features.

### 6. Extract chat-faithful activations

Configured layer numbers are zero-based transformer-block outputs, not indices into the Hugging Face `hidden_states` tuple.

```bash
uv run python -m cli.extract_task_activations \
  --task sycophancy \
  --data data/labeled/Qwen3-4B/sycophancy.jsonl \
  --model Qwen/Qwen3-4B \
  --model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --layers 7,15,23,31 \
  --views answer \
  --output_dir outputs/frontier_features/Qwen3-4B/sycophancy \
  --device mps

uv run python -m cli.extract_task_activations \
  --task benign_calibration \
  --data data/labeled/Qwen3-4B/benign_calibration.jsonl \
  --model Qwen/Qwen3-4B \
  --model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --layers 7,15,23,31 \
  --views answer \
  --calibration_only \
  --output_dir outputs/frontier_features/Qwen3-4B/benign_calibration \
  --device mps
```

Extraction uses the tokenizer's chat template, respects declared group-disjoint splits, and fails if any requested span is absent or truncated. The independently screened benign dataset is extracted as calibration-only.

### 7. Run black-box leakage baselines

```bash
uv run python -m cli.run_text_baselines \
  --source_task sycophancy \
  --source_data data/labeled/Qwen3-4B/sycophancy.jsonl \
  --target_task motivated_reasoning \
  --target_data data/labeled/Qwen3-4B/motivated_reasoning.jsonl \
  --calibration_task benign_calibration \
  --calibration_data data/labeled/Qwen3-4B/benign_calibration.jsonl \
  --model Qwen3-4B \
  --results_dir results/frontier_pilot \
  --min_calibration_negatives 1000
```

The stronger frozen encoder baseline uses a separately pinned Qwen3 embedding model. Each dataset/view is embedded once; every few-shot seed reuses the same immutable cache.

```bash
uv run python -m cli.extract_text_embeddings \
  --task sycophancy \
  --data data/labeled/Qwen3-4B/sycophancy.jsonl \
  --embedding_config experiments/baselines/text_embedding_models.yaml \
  --embedding_model_key qwen3_embedding_0_6b \
  --output_dir outputs/text_embeddings/Qwen3-4B/sycophancy

uv run python -m cli.extract_text_embeddings \
  --task motivated_reasoning \
  --data data/labeled/Qwen3-4B/motivated_reasoning.jsonl \
  --embedding_config experiments/baselines/text_embedding_models.yaml \
  --embedding_model_key qwen3_embedding_0_6b \
  --output_dir outputs/text_embeddings/Qwen3-4B/motivated_reasoning

uv run python -m cli.extract_text_embeddings \
  --task benign_calibration \
  --data data/labeled/Qwen3-4B/benign_calibration.jsonl \
  --embedding_config experiments/baselines/text_embedding_models.yaml \
  --embedding_model_key qwen3_embedding_0_6b \
  --output_dir outputs/text_embeddings/Qwen3-4B/benign_calibration

uv run python -m cli.run_embedding_baselines \
  --source_task sycophancy \
  --source_cache_dir outputs/text_embeddings/Qwen3-4B/sycophancy \
  --target_task motivated_reasoning \
  --target_cache_dir outputs/text_embeddings/Qwen3-4B/motivated_reasoning \
  --calibration_task benign_calibration \
  --calibration_cache_dir outputs/text_embeddings/Qwen3-4B/benign_calibration \
  --model Qwen3-4B \
  --results_dir results/frontier_pilot \
  --min_calibration_negatives 1000
```

The independent LLM judge is pinned to an exact model and tokenizer commit. It scores only the registered A/B next-token logits, prohibits truncation, treats the interaction as untrusted text, and stores content-hashed score caches so interrupted runs resume without rescoring completed contexts.

```bash
uv run python -m cli.run_llm_judge_baselines \
  --source_task sycophancy \
  --source_data data/labeled/Qwen3-4B/sycophancy.jsonl \
  --target_task motivated_reasoning \
  --target_data data/labeled/Qwen3-4B/motivated_reasoning.jsonl \
  --calibration_task benign_calibration \
  --calibration_data data/labeled/Qwen3-4B/benign_calibration.jsonl \
  --judge_config experiments/baselines/llm_judge_models.yaml \
  --judge_model_key phi4_14b \
  --judge_cache_dir outputs/llm_judge_scores/Qwen3-4B \
  --model Qwen3-4B \
  --results_dir results/frontier_pilot \
  --min_calibration_negatives 1000
```

The output-confidence baseline learns from the generation-time log-probability, entropy, top-1 margin, and length summaries stored in each rollout. It refuses legacy rollouts without aligned confidence traces.

```bash
uv run python -m cli.run_output_confidence_baselines \
  --source_task sycophancy \
  --source_data data/labeled/Qwen3-4B/sycophancy.jsonl \
  --target_task motivated_reasoning \
  --target_data data/labeled/Qwen3-4B/motivated_reasoning.jsonl \
  --calibration_task benign_calibration \
  --calibration_data data/labeled/Qwen3-4B/benign_calibration.jsonl \
  --model Qwen3-4B \
  --results_dir results/frontier_pilot \
  --min_calibration_negatives 1000
```

All matched few-shot baselines use the same group-aware label budget, fit the operating threshold only on dedicated benign calibration negatives, save per-example predictions, and apply that threshold unchanged to source and transfer tests. The zero-shot judge is scored once per view and reused across paired seed comparisons without rerunning the model.

### 8. Run activation monitors

```bash
uv run python -m cli.run_task_sweep \
  --source_dir outputs/frontier_features/Qwen3-4B/sycophancy \
  --source_task sycophancy \
  --calibration_dir outputs/frontier_features/Qwen3-4B/benign_calibration \
  --target_dir outputs/frontier_features/Qwen3-4B/motivated_reasoning \
  --target_task motivated_reasoning \
  --model Qwen3-4B \
  --results_dir results/frontier_pilot \
  --views answer \
  --layers all \
  --probes P1_logistic,P2_mass_mean \
  --k_values 1,2,4,8 \
  --seeds 10 \
  --balance_modes balanced \
  --min_calibration_negatives 1000
```

The sweep writes one atomic prediction artifact per run. Unsupported probe/sample combinations are not treated as evidence.

### 9. Run pre-registered paired inference

Exact systems must be declared before final-test inspection:

```bash
uv run python -m cli.build_frozen_transfer_report \
  --results_dir results/frontier_pilot \
  --selection_k 8

uv run python -m cli.compute_task_significance \
  --results_dir results/frontier_pilot \
  --comparisons experiments/protocol/preregistered_comparisons.example.yaml \
  --bootstrap_samples 5000

uv run python -m cli.compute_falsification_significance \
  --results_dir results/frontier_pilot \
  --registry experiments/protocol/falsification_registry.yaml \
  --comparisons experiments/protocol/preregistered_falsification_comparisons.example.yaml \
  --bootstrap_samples 5000
```

The selector writes `task_primary_source_systems.csv`, with one family, balance mode, layer, and view identity per access regime. A comparison marked `primary_white_box_gain` must match those identities exactly or inference stops. For a frozen run, register one primary comparison for every configured model, task pair, and label budget. The multi-model manifest records the completed files as `comparisons_file` and `falsification_comparisons_file`. Both inference paths pair identical evidence across systems, jointly resample few-shot seeds and independent scenario groups, and report system-A-minus-system-B differences. Falsification hypotheses share one global Holm family; the comparison files and their hashes are archived before tables are emitted.

## Final execution gate

```bash
uv run python -m cli.validate_multimodel_config \
  --config experiments/protocol/frozen_frontier_manifest.yaml \
  --check_paths \
  --final_protocol
```

A final run is blocked unless it has a frozen protocol, at least three model families, at least 10,000 independently screened calibration groups, ten training seeds, immutable revisions, complete activation/embedding provenance, confidence provenance wherever source artifacts expose genuine generation-time distributions, all three visible-text views, the TF-IDF, frozen-embedding, zero/few-shot judge, and output-confidence baselines on supported pairs, an independent frozen-primary judge family, and existing pre-registered primary and falsification comparison files. It also requires registry-bound held-out coverage for all five shift axes, an inferential comparison for every axis and registered behavior transfer, at least 100 independent groups per held-out axis, both labels in every shift slice, and at least 100 independent exact-prompt hard-negative groups plus a registered comparison per enabled task. Repeated rollouts from one benign prompt do not inflate any count. A task-level B4 inapplicability is frozen in its task contract and is never replaced with reconstructed confidence.

### Frontier one-command runner

```bash
uv run python -m cli.run_frontier_suite --config experiments/protocol/neurips_main_manifest.yaml --device mps
```

Use `--dry-run` to preview the command graph, `--device mps` on Apple Silicon, and `--release-artifacts` when running a frozen manifest.

### Run the full frontier experiment portfolio

```bash
uv run python -m cli.run_frontier_experiments
```

This executes:

- main frontier protocol (`experiments/protocol/neurips_main_manifest.yaml`)
- main ablations (`experiments/protocol/ablation_suite.yaml`)
- optional negative controls
- optional honesty-control auxiliary run (`experiments/controls/honesty_auxiliary_manifest.yaml`)

Flags:

- `--no-honesty` to skip the auxiliary task.
- `--include-honesty-controls` to run controls on the auxiliary manifest as well.
- `--run-appendix-ablations --appendix-ablation-config <path>` for appendix-style ablations.
- `--device mps` for Apple Silicon.
- `--dry-run` and `--skip-validation` for safe planning.

### New-machine setup

```bash
python -m pip install uv
uv sync --locked --extra dev
uv run python -m cli.validate_multimodel_config --config experiments/protocol/neurips_main_manifest.yaml --check_paths
```

On a fresh machine, ensure all required source artifacts are available (the runner now stops early if declared inputs are missing) and set your Hugging Face credentials before any fetch/rollout steps.

```bash
export HF_HOME=...
export HF_TOKEN=...
```

Use one command to force model-backed execution onto MPS. If memory is tight, reduce the registered batch size or run one model at a time; do not change layers, context limits, or model identities mid-run.

```bash
uv run python -m cli.run_frontier_experiments --device mps
```

## Legacy warning

Legacy authored-completion workflows are intentionally non-blocking for the active frontier stack and are not required for current execution.

Pinned raw-source metadata remains in `experiments/data/huggingface_source_lock.yaml`; MonitorBench is locked to a commit, archive checksum, critical-file hashes, and complete extracted-tree manifest rather than a moving branch.
