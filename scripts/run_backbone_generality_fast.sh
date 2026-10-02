#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SEED=20260928

BASE_CKPT="experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/best_model_results/visual_jev_v3_best.pt"

declare -A MODELS
MODELS[siglip2]="models/SigLIP2-Base-Patch16-224"
MODELS[internvl]="models/InternVL3_5-1B-HF"
MODELS[llava]="models/LLaVA-OneVision-0.5B"

echo "========================================"
echo "Checking downloaded models"
echo "========================================"

for name in siglip2 internvl llava; do
    path="${MODELS[$name]}"

    if [[ ! -d "$path" ]]; then
        echo "[ERROR] Missing model: $path"
        exit 1
    fi

    if [[ ! -f "$path/config.json" ]]; then
        echo "[ERROR] Incomplete model: $path"
        exit 1
    fi

    echo "[OK] $name -> $path"
done

echo
echo "All models found."
echo

for name in siglip2 internvl llava; do

    path="${MODELS[$name]}"

    echo
    echo "============================================================"
    echo "Running backbone: $name"
    echo "============================================================"

    "$PYTHON" scripts/run_backbone_adapter_fast.py \
        --backbone "$name" \
        --model "$path" \
        --base-checkpoint "$BASE_CKPT" \
        --manifest-root data/benchmark_v1_full/manifests \
        --qwen-feature-root experiments/benchmark_v1_full/features \
        --dataset-substr coco \
        --device "$DEVICE" \
        --seed "$SEED" \
        --train-limit 512 \
        --validation-limit 80 \
        --calibration-limit 80 \
        --test-limit 80 \
        --epochs 3 \
        --learning-rate 3e-4 \
        --grad-accumulation 8 \
        --output "experiments/results/backbone_generality/$name" \
        2>&1 | tee "logs/backbone_${name}.log"

done

echo
echo "============================================================"
echo "Backbone generality experiments completed."
echo "============================================================"

"$PYTHON" scripts/summarize_backbone_generality.py \
    experiments/results/backbone_generality

