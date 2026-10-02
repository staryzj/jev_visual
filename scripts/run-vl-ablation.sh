#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPEN_JEV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL:-$ROOT/models/Qwen3-VL-4B-Instruct}"
DATA="${DATA:-$ROOT/data/vl-paper}"
IMAGE_ROOT="${IMAGE_ROOT:-$ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs/vl-paper}"
STEPS="${STEPS:-1000}"
TRAIN_ROWS="${TRAIN_ROWS:-0}"
CALIBRATION_ROWS="${CALIBRATION_ROWS:-0}"
EVAL_ROWS="${EVAL_ROWS:-0}"
ACCUMULATION="${ACCUMULATION:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-250}"
SEEDS="${SEEDS:-42 43 44}"

if [[ ! -f "$MODEL/config.json" || ! -f "$DATA/train.jsonl" ]]; then
  echo "Missing local model or VL dataset. MODEL=$MODEL DATA=$DATA" >&2
  exit 2
fi

for seed in $SEEDS; do
  for specification in "head:0" "language-lora:8" "vision-language-lora:8"; do
    mode="${specification%%:*}"
    rank="${specification##*:}"
    output="$OUTPUT_ROOT/${mode}-r${rank}-seed${seed}"
    "$ROOT/.venv/bin/python" -m jev.train \
      --model "$MODEL" \
      --vision \
      --image-root "$IMAGE_ROOT" \
      --data "$DATA" \
      --output "$output" \
      --vl-tuning "$mode" \
      --lora-rank "$rank" \
      --steps "$STEPS" \
      --train-rows "$TRAIN_ROWS" \
      --calibration-rows "$CALIBRATION_ROWS" \
      --eval-rows "$EVAL_ROWS" \
      --accumulation "$ACCUMULATION" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      --max-length 2048 \
      --min-pixels 200704 \
      --max-pixels 401408 \
      --seed "$seed"
  done
done
