#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_extract() {
  local model="$1"
  local dataset="$2"
  "$PYTHON_BIN" extract_activations.py --config "$CONFIG" --model "$model" --dataset "$dataset" --mode all
}

run_extract "Qwen/Qwen3-4B" "enron"
run_extract "Qwen/Qwen3-4B" "sms"
run_extract "Qwen/Qwen3-4B" "mask"

run_extract "Qwen/Qwen3-8B" "enron"
run_extract "Qwen/Qwen3-8B" "sms"
run_extract "Qwen/Qwen3-8B" "mask"

run_extract "google/gemma-2-9b" "enron"
run_extract "google/gemma-2-9b" "sms"
run_extract "google/gemma-2-9b" "mask"

"$PYTHON_BIN" run_sweep.py --config "$CONFIG" --probes \
  P1_logistic P2_mass_mean P3_lda P4_cosine P6_prompted P7_mahalanobis P8_followup

"$PYTHON_BIN" analyze.py --config "$CONFIG"
"$PYTHON_BIN" significance_runner.py --config "$CONFIG" --split test
"$PYTHON_BIN" plot_runner.py --config "$CONFIG"

echo "Lane A benchmark completed."
