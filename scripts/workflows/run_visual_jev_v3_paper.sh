#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ ! -s data/visual-jev-v3/pairs/train.jsonl || ! -s data/visual-jev-v3/pairs/validation.jsonl ]]; then
  .venv/bin/python scripts/data/prepare_visual_jev_v3_pairs.py
fi

.venv/bin/python scripts/training/train_visual_jev_v3.py --config configs/visual_jev_v3_paper.json
.venv/bin/python scripts/evaluation/eval_visual_jev_v3.py --config configs/visual_jev_v3_paper.json --native
.venv/bin/python scripts/evaluation/diagnostic_visual_dependency.py --config configs/visual_jev_v3_paper.json
.venv/bin/python scripts/data/prepare_benchmark_v1.py
.venv/bin/python scripts/evaluation/calibrate_visual_jev_v3.py --config configs/visual_jev_v3_paper.json

echo "Visual-JEV V3 paper experiment completed."
echo "Results: experiments/results/metrics.json"
echo "Calibration: experiments/results/calibration.json"
echo "Benchmark manifest: data/benchmark_v1/manifests/manifest.json"
echo "Report: docs/visual_jev_v3_experiment.md"
