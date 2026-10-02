#!/usr/bin/env bash
# Historical custom-protocol full training; NEVER relabel its held-out set as official test.
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# )); then
  echo 'This reproducibility workflow fixes full training with no cap; invoke run_independent_datasets.py directly for another explicit protocol.' >&2
  exit 2
fi
.venv/bin/python run_independent_datasets.py \
  --train-cap 0 \
  --result-root experiments/results/independent_datasets_full_reproduction \
  --resume
