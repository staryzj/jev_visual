#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# )); then
  echo 'For targeted/custom-path recovery use scripts/resume_submission_benchmark.py; this bulk workflow accepts no arguments.' >&2
  exit 2
fi
.venv/bin/python scripts/submission_benchmarks.py prepare --fetch-images
.venv/bin/python scripts/evaluate_submission_benchmarks.py --controls --repair-wrong-controls
.venv/bin/python scripts/generate_submission_report.py
