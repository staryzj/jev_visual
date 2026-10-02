#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/results/benchmark_v1/fix_negative_transfer_server}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$LOG_DIR/negative-transfer-$RUN_ID.log"

mkdir -p "$LOG_DIR" "$RESULTS_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  echo "Create it with: uv sync --extra train --extra vl" >&2
  exit 2
fi

required=(
  "data/benchmark_v1/manifests/train.jsonl"
  "data/visual-jev-v2/splits/train.jsonl"
  "data/visual-jev-v3/pairs/train.jsonl"
  "experiments/benchmark_v1_controlled/features/fixed_controls.pt"
  "experiments/visual_jev_v3_paper/checkpoints/v3_full.pt"
)
missing=()
for path in "${required[@]}"; do
  [[ -e "$path" ]] || missing+=("$path")
done
if ((${#missing[@]})); then
  echo "Required training resources are missing:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  echo "See SERVER_RUN.md for the expected resource layout." >&2
  exit 3
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
  echo "Warning: nvidia-smi is unavailable; CUDA training may fail." >&2
fi

export PYTHONUNBUFFERED=1
echo "Training log: $LOG_PATH"
echo "Progress bars show step/sample progress, speed and ETA. Epoch summaries include total/domain loss and validation metrics."

"$PYTHON_BIN" run_fix_negative_transfer.py \
  --device "$DEVICE" \
  --results-root "$RESULTS_ROOT" 2>&1 | tee "$LOG_PATH"

BEST_CHECKPOINT="$RESULTS_ROOT/best_model_results/visual_jev_v3_best.pt"
if [[ ! -f "$BEST_CHECKPOINT" ]]; then
  echo "Training completed without the expected best checkpoint: $BEST_CHECKPOINT" >&2
  exit 4
fi

"$PYTHON_BIN" run_calibration_comparison.py \
  --checkpoint "$BEST_CHECKPOINT" \
  --results-root "$RESULTS_ROOT/calibration_after_fix" \
  --device "$DEVICE" 2>&1 | tee -a "$LOG_PATH"

echo "Completed. Results: $RESULTS_ROOT"
echo "Best checkpoint: $BEST_CHECKPOINT"
echo "Full log: $LOG_PATH"

