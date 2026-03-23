# Probe Benchmark: Which Probe Should You Use When Positives Are Scarce?

Head-to-head comparison of 8 activation probe architectures for few-shot safety monitoring. Tests which probe works best when you have 1 to 100 positive examples.

Based on Tyagi and Heimersheim (NeurIPS 2025 Workshop).

## Probes tested

| ID | Probe | Method |
|----|-------|--------|
| P1 | Logistic Regression | L2-regularised linear classifier on mean-pooled activations |
| P2 | Mass-Mean | Projection onto class mean difference vector |
| P3 | LDA | Fisher's linear discriminant with shrinkage |
| P4 | Cosine Similarity | Cosine distance to class centroids |
| P5 | SAE Probe | Sparse autoencoder features + L1 logistic regression (Gemma only) |
| P6 | Prompted Probing | Logistic regression on activations from task-prompted input |
| P7 | Mahalanobis Distance | Anomaly score from negative-class distribution |
| P8 | Follow-up Question | Logistic regression on activations after appending a deception question |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

The pipeline has three phases. Each one depends on the previous.

### Phase 1: Extract activations (GPU)

Runs one forward pass per sample through the model and caches hidden states to disk as `.npy` files. This is the only phase that needs a GPU.

```bash
# Extract all three activation modes (standard, prompted, followup) for a model+dataset pair
python extract_activations.py --model Qwen/Qwen3-4B --dataset enron --mode all
python extract_activations.py --model Qwen/Qwen3-4B --dataset sms --mode all
```

`--mode` options: `standard`, `prompted`, `followup`, or `all` (runs all three).

Repeat for each model you want to benchmark (`Qwen/Qwen3-8B`, `google/gemma-2-9b`).

Cached activations go to `./cache/activations/{model}/{dataset}/`. Expect 10-20 GB total across all models and datasets.

### Phase 2: Run the benchmark sweep (CPU)

Trains every probe at every k value, both training modes, 10 seeds. Reads from the activation cache -- no GPU needed.

```bash
python run_sweep.py --config config.yaml
```

Optional filters to run a subset:

```bash
python run_sweep.py --config config.yaml --model Qwen/Qwen3-4B --dataset enron --probes P1_logistic P2_mass_mean
```

Results are written as JSONL files to `./results/`.

### Phase 3: Analyze (CPU)

Aggregates results, picks best layers, computes FSEI, and generates the decision table.

```bash
python analyze.py --config config.yaml
```

Outputs in `./results/`:
- `summary.csv` -- mean/std metrics across seeds for every configuration
- `best_layer_summary.csv` -- results at the best layer per probe
- `fsei.csv` -- Few-Shot Efficiency Index ranking
- `decision_table.csv` -- best probe per k value

## Configuration

All experiment parameters live in `config.yaml`:

- **models**: which models and which layers to probe
- **datasets**: HuggingFace dataset IDs
- **extraction**: batch size, max sequence length, pooling strategy
- **sweep**: k values, number of seeds, balance modes
- **filtering**: logit difference threshold for confidence filtering

## Project structure

```
.
├── config.yaml              # Experiment configuration
├── extract_activations.py   # Phase 1: GPU activation extraction
├── run_sweep.py             # Phase 2: CPU benchmark sweep
├── analyze.py               # Phase 3: Results analysis
├── data/
│   ├── loading.py           # Dataset loaders (Enron, SMS)
│   ├── filtering.py         # Logit confidence filtering
│   └── splitting.py         # Train/eval/test splits, k-shot sampling
├── extraction/
│   ├── extractor.py         # Standard activation extraction + caching
│   └── modified_extractor.py # Prompted (P6) and followup (P8) extraction
├── probes/
│   ├── base.py              # Abstract Probe interface
│   ├── logistic.py          # P1: Logistic Regression
│   ├── mass_mean.py         # P2: Mass-Mean Probe
│   ├── lda.py               # P3: LDA
│   ├── cosine.py            # P4: Cosine Similarity
│   ├── sae_probe.py         # P5: SAE Probe
│   ├── prompted.py          # P6: Prompted Probing
│   ├── mahalanobis.py       # P7: Mahalanobis Distance
│   └── followup.py          # P8: Follow-up Question Probe
├── evaluation/
│   ├── metrics.py           # AUROC, Recall@1%FPR, FSEI
│   └── aggregation.py       # Result collection, summary stats, decision table
└── requirements.txt
```

## Compute requirements

No model training or fine-tuning. GPU is only needed for activation extraction.

- **Minimum**: L4 24 GB or equivalent (Colab Pro, Lightning AI). All three models fit at full precision.
- **Probe training**: CPU only after the cache is built.
- **Estimated cost**: Under $20 total across both platforms.

## Datasets

- **Training**: [Enron-Spam](https://huggingface.co/datasets/SetFit/enron_spam) -- off-policy spam classification
- **OOD test**: [SMS Spam](https://huggingface.co/datasets/ucirvine/sms_spam) -- never seen during training, tests distribution shift
