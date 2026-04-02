# Few Shot Activation Probe Benchmark

This repository benchmarks activation probe families in the extreme positive scarcity regime.

## Main question

When only a handful of positive examples are available, which probe family should you try first?

## Probe families in the main benchmark

| ID | Probe | Method |
|----|-------|--------|
| P1 | Logistic Regression | L2-regularized linear classifier on mean-pooled activations |
| P2 | Mass-Mean | Projection onto the class mean difference vector |
| P3 | LDA | Fisher's linear discriminant with shrinkage |
| P4 | Cosine Similarity | Cosine distance to class centroids |
| P6 | Prompted Probing | Logistic regression on activations from task-prompted input |
| P7 | Mahalanobis Distance | Anomaly score from the negative-class distribution |
| P8 | Followup Context Probing | Logistic regression on activations from the concatenated followup context |

`P5_sae` is excluded from the main benchmark in the current workshop version because a fully faithful pretrained SAE pipeline is not yet wired end to end. The file [sae_probe.py](https://github.com/rohanpoudel2/probe/blob/main/probes/sae_probe.py) remains in the repo but is not part of the default run path.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
uv venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

All commands below assume the environment is active. If you prefer, you can also run them with `uv run`.

## Supported datasets

- `enron`
- `sms`
- `mask`

The benchmark treats Enron and MASK as in-distribution tasks. SMS is used as an out-of-distribution transfer target for Enron. The MASK loader targets the official `cais/MASK` dataset and normalizes around proposition, ground truth, pressure prompt, and belief elicitation prompt fields.

## Experiment protocol

1. Extract and cache activations for each model, dataset, and extraction mode.
2. Filter examples using task-aware logit difference prompts.
3. Build reproducible `train_pool`, `eval`, and `test` splits.
4. Sample `k` positive examples from `train_pool` and fit probes.
5. Select the best layer using `eval` performance.
6. Report final metrics on `test`.
7. Report OOD transfer where available.
8. Run paired bootstrap significance testing on the top two probes per setting.

## Main metrics

- AUROC
- Recall at 1 percent false positive rate
- Few-Shot Efficiency Index

## How to run

The pipeline has five phases. Each one depends on the previous.

### Phase 1: Extract activations (GPU)

Runs one forward pass per sample through the model and caches hidden states to disk as `.npy` files. This is the only phase that needs a GPU.

```bash
python extract_activations.py --config config.yaml --model "Qwen/Qwen3-4B" --dataset enron --mode all
python extract_activations.py --config config.yaml --model "Qwen/Qwen3-4B" --dataset sms --mode all
python extract_activations.py --config config.yaml --model "Qwen/Qwen3-4B" --dataset mask --mode all
```

`--mode` options: `standard`, `prompted`, `followup`, or `all`.

### Phase 2: Run the benchmark sweep (CPU)

Trains every main-benchmark probe at every `k`, both balance modes, across all seeds. Reads from the activation cache and does not need a GPU.

```bash
python run_sweep.py --config config.yaml --probes \
  P1_logistic P2_mass_mean P3_lda P4_cosine P6_prompted P7_mahalanobis P8_followup
```

Optional subset example:

```bash
python run_sweep.py --config config.yaml --model "Qwen/Qwen3-4B" --dataset enron --probes P1_logistic P2_mass_mean
```

### Phase 3: Analyze benchmark outputs (CPU)

Aggregates raw rows, computes summary statistics, chooses best layers, and writes the main paper tables.

```bash
python analyze.py --config config.yaml
```

### Phase 4: Run significance testing (CPU)

```bash
python significance_runner.py --config config.yaml --split test
```

This writes `significance_test.csv`. If you run `--split eval`, it writes `significance_eval.csv`.

### Phase 5: Generate plots (CPU)

```bash
python plot_runner.py --config config.yaml
```

### Full run

```bash
bash scripts/run_full_benchmark.sh
```

If you want to avoid activating the environment explicitly, the equivalent `uv` style is:

```bash
uv run python extract_activations.py --config config.yaml --model "Qwen/Qwen3-4B" --dataset enron --mode all
uv run python run_sweep.py --config config.yaml --probes P1_logistic P2_mass_mean P3_lda P4_cosine P6_prompted P7_mahalanobis P8_followup
uv run python analyze.py --config config.yaml
uv run python significance_runner.py --config config.yaml --split test
uv run python plot_runner.py --config config.yaml
```

## Key outputs

After running the benchmark, the results directory contains:

- `summary.csv`
- `best_layer_summary.csv`
- `decision_table.csv`
- `ood_table.csv`
- `fsei.csv`
- `layer_choices.csv`
- `significance_test.csv`
- `figures/`

## Configuration

All experiment parameters live in [config.yaml](https://github.com/rohanpoudel2/probe/blob/main/config.yaml):

- `models`: which models and layers to probe
- `datasets`: in-distribution tasks and OOD transfer mapping
- `extraction`: batch size, max sequence length, cache directory, pooling mode
- `sweep`: `k` values, number of seeds, balance modes
- `filtering`: logit-difference threshold for confidence filtering
- `selection_metric`: metric used for best-layer choice
- `results`: overwrite and prediction-saving behavior
- `significance`: bootstrap settings

## Project structure

```text
.
|-- config.yaml                 # Experiment configuration
|-- extract_activations.py      # Phase 1: GPU activation extraction
|-- run_sweep.py                # Phase 2: CPU benchmark sweep
|-- analyze.py                  # Phase 3: summary tables and best-layer reporting
|-- significance_runner.py      # Phase 4: paired bootstrap significance testing
|-- plot_runner.py              # Phase 5: plot generation entrypoint
|-- data/
|   |-- loading.py              # Dataset loaders (Enron, SMS, MASK)
|   |-- filtering.py            # Task-aware logit confidence filtering
|   `-- splitting.py            # Train/eval/test splits and k-shot sampling
|-- extraction/
|   |-- extractor.py            # Standard activation extraction and caching
|   `-- modified_extractor.py   # Prompted (P6) and followup-context (P8) extraction
|-- probes/
|   |-- base.py                 # Abstract probe interface
|   |-- logistic.py             # P1: Logistic Regression
|   |-- mass_mean.py            # P2: Mass-Mean Probe
|   |-- lda.py                  # P3: LDA
|   |-- cosine.py               # P4: Cosine Similarity
|   |-- prompted.py             # P6: Prompted Probing
|   |-- mahalanobis.py          # P7: Mahalanobis Distance
|   |-- followup.py             # P8: Followup Context Probing
|   `-- sae_probe.py            # P5: SAE Probe (not in main benchmark)
|-- evaluation/
|   |-- metrics.py              # AUROC, Recall@1%FPR, FSEI
|   |-- aggregation.py          # Summary stats, layer selection, decision tables
|   |-- significance.py         # Bootstrap comparison utilities
|   `-- plots.py                # Figure generation helpers
|-- scripts/
|   `-- run_full_benchmark.sh   # End-to-end Lane A benchmark run
`-- requirements.txt
```

## Compute requirements

No model training or fine-tuning. GPU is only needed for activation extraction.

- Minimum: L4 24 GB or equivalent
- Probe training: CPU only after the cache is built
- The safest first run is a small smoke test before full extraction

## Recommended smoke test

Run these before a full benchmark:

```bash
python extract_activations.py --config config.yaml --model "Qwen/Qwen3-4B" --dataset mask --mode standard
python run_sweep.py --config config.yaml --model "Qwen/Qwen3-4B" --dataset enron --probes P1_logistic P2_mass_mean
python analyze.py --config config.yaml
python significance_runner.py --config config.yaml --split test
python plot_runner.py --config config.yaml
```
 
## Datasets

- Training / ID task 1: [Enron-Spam](https://huggingface.co/datasets/SetFit/enron_spam)
- Training / ID task 2: [MASK](https://huggingface.co/datasets/cais/MASK)
- OOD transfer: [SMS Spam](https://huggingface.co/datasets/ucirvine/sms_spam)

## Current limitations

- The exact exposed Hugging Face field names for `cais/MASK` should still be checked once in your runtime environment.
- `P8_followup` currently pools over the full concatenated context rather than followup tokens only, so it should be interpreted as a context-based augmentation baseline rather than a pure followup token readout.
