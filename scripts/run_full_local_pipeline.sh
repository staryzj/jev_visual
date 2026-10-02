#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  echo "Activate conda and run with PYTHON_BIN=\"$CONDA_PREFIX/bin/python\"." >&2
  exit 1
fi

mkdir -p logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[1/4] Building leakage-safe full manifests..."
if [[ -s data/benchmark_v1_full/manifests/manifest.json \
   && -s data/benchmark_v1_full/manifests/train.jsonl \
   && -s data/benchmark_v1_full/manifests/validation.jsonl \
   && -s data/benchmark_v1_full/manifests/calibration.jsonl \
   && -s data/benchmark_v1_full/manifests/test.jsonl ]]; then
  echo "Full manifests already exist; reusing them."
else
  "$PYTHON_BIN" prepare_benchmark_v1.py \
    --raw-root data/visual_jev_raw \
    --output data/benchmark_v1_full/manifests \
    --seed 20260928 \
    --full \
    2>&1 | tee -a logs/01_prepare_full_manifest.log
fi

echo "[2/4] Extracting frozen Qwen3-VL features (resumable)..."
"$PYTHON_BIN" run_benchmark_v1_experiment.py \
  --manifest-root data/benchmark_v1_full/manifests \
  --experiment-root experiments/benchmark_v1_full \
  --results-root experiments/results/benchmark_v1/full_feature_preparation \
  --model models/Qwen3-VL-4B-Instruct \
  --device cuda:0 \
  --prepare-only \
  2>&1 | tee -a logs/02_prepare_full_features.log

echo "[3/4] Training every eligible training sample..."
"$PYTHON_BIN" train_negative_transfer_full.py \
  --manifest-root data/benchmark_v1_full/manifests \
  --experiment-root experiments/benchmark_v1_full \
  --initial-checkpoint experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/best_model_results/visual_jev_v3_best.pt \
  --output experiments/results/benchmark_v1/fix_negative_transfer/full_run \
  --device cuda:0 \
  --candidate-epochs 2 \
  --stagec-epochs 3 \
  --easy-stagec-epochs 1 \
  --learning-rate 2e-5 \
  --grad-accumulation 8 \
  --pair-interval 8 \
  2>&1 | tee -a logs/03_full_training.log

echo "[4/4] Testing and fitting calibration on calibration split only..."
"$PYTHON_BIN" run_calibration_comparison.py \
  --manifest-root data/benchmark_v1_full/manifests \
  --experiment-root experiments/benchmark_v1_full \
  --checkpoint experiments/results/benchmark_v1/fix_negative_transfer/full_run/best.pt \
  --results-root experiments/results/benchmark_v1/fix_negative_transfer/calibration_after_fix_full \
  --device cuda:0 \
  --min-domain-samples 8 \
  2>&1 | tee -a logs/04_full_test_and_calibration.log

echo "Full pipeline completed successfully."
