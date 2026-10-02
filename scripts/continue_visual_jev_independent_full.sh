#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
FULL_MANIFEST_ROOT="${FULL_MANIFEST_ROOT:-$PROJECT_ROOT/data/benchmark_v1_full/manifests}"
FULL_EXPERIMENT_ROOT="${FULL_EXPERIMENT_ROOT:-$PROJECT_ROOT/experiments/benchmark_v1_full}"
CONTROLLED_MANIFEST_ROOT="${CONTROLLED_MANIFEST_ROOT:-$PROJECT_ROOT/data/benchmark_v1/manifests}"
CONTROLLED_EXPERIMENT_ROOT="${CONTROLLED_EXPERIMENT_ROOT:-$PROJECT_ROOT/experiments/benchmark_v1_controlled}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-$PROJECT_ROOT/experiments/visual_jev_v3_paper/checkpoints/v3_full.pt}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/experiments/results/independent_datasets_full}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/independent_datasets}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-2}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-16}"
STAGE_C_EPOCHS="${STAGE_C_EPOCHS:-6}"
CHECK_ONLY=0
RESUME=1

usage() {
  cat <<'EOF'
Usage: scripts/continue_visual_jev_independent_full.sh [--check-only] [--no-resume]

Runs four full-manifest adaptations independently while reusing the completed
frozen Qwen3-VL feature cache. Environment variables can override PYTHON_BIN,
DEVICE, RESULT_ROOT, DATA_ROOT, and STAGE_{A,B,C}_EPOCHS.
EOF
}

while (($#)); do
  case "$1" in
    --check-only) CHECK_ONLY=1 ;;
    --no-resume) RESUME=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$LOG_DIR" "$RESULT_ROOT"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/independent-full-$RUN_ID.log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'status=$?; echo "[failed] exit=$status line=$LINENO log=$LOG_FILE"; exit "$status"' ERR

echo "[start] $(date --iso-8601=seconds)"
echo "[log] $LOG_FILE"
echo "[mode] four independent datasets; train-cap=0; frozen caches only"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$PROJECT_ROOT/run_independent_datasets.py" ]]; then
  echo "Missing independent runner: $PROJECT_ROOT/run_independent_datasets.py" >&2
  exit 1
fi
for value in "$STAGE_A_EPOCHS" "$STAGE_B_EPOCHS" "$STAGE_C_EPOCHS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Stage epoch counts must be positive integers; got: $value" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - \
  "$FULL_MANIFEST_ROOT" \
  "$FULL_EXPERIMENT_ROOT" \
  "$CONTROLLED_MANIFEST_ROOT" \
  "$CONTROLLED_EXPERIMENT_ROOT" \
  "$REFERENCE_CHECKPOINT" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

full_manifest_root = Path(sys.argv[1])
full_experiment_root = Path(sys.argv[2])
controlled_manifest_root = Path(sys.argv[3])
controlled_experiment_root = Path(sys.argv[4])
reference_checkpoint = Path(sys.argv[5])
splits = ("train", "validation", "calibration", "test")
datasets = {
    "aokvqa",
    "coco_hard_negative",
    "iconqa_choice",
    "scienceqa_image_only",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

required = [
    full_manifest_root / "manifest.json",
    full_experiment_root / "features" / "fixed_controls.pt",
    controlled_experiment_root / "features" / "fixed_controls.pt",
    reference_checkpoint,
    Path("data/visual-jev-v2/splits/train.jsonl"),
    Path("data/visual-jev-v2/splits/validation.jsonl"),
    Path("data/visual-jev-v3/pairs/train.jsonl"),
    Path("data/visual-jev-v3/pairs/validation.jsonl"),
]
required.extend(controlled_manifest_root / f"{split}.jsonl" for split in splits)
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit("missing required input(s):\n  " + "\n  ".join(missing))

summary = {}
for split in splits:
    source = full_manifest_root / f"{split}.jsonl"
    if not source.is_file():
        raise SystemExit(f"missing full manifest split: {source}")
    counts = Counter()
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                counts[json.loads(line)["dataset"]] += 1
    if set(counts) != datasets or any(value <= 0 for value in counts.values()):
        raise SystemExit(f"{split}: expected exactly four non-empty datasets, got {dict(counts)}")
    cache = full_experiment_root / "features" / f"{split}-shards"
    cache_manifest = cache / "manifest.json"
    if not cache_manifest.is_file():
        raise SystemExit(f"missing cache manifest: {cache_manifest}")
    payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
    expected_count = sum(counts.values())
    if int(payload.get("count", -1)) != expected_count:
        raise SystemExit(f"{split}: cache count {payload.get('count')} != manifest count {expected_count}")
    source_hash = sha256(source)
    if payload.get("source_sha256") != source_hash:
        raise SystemExit(f"{split}: frozen cache source hash is stale")
    shard_paths = list(cache.glob("*.pt"))
    shard_names = {path.name for path in shard_paths}
    expected_names = {f"{index:06d}.pt" for index in range(expected_count)}
    if shard_names != expected_names:
        absent = sorted(expected_names - shard_names)[:5]
        extra = sorted(shard_names - expected_names)[:5]
        raise SystemExit(f"{split}: shard set mismatch; missing={absent}, extra={extra}")
    empty = sorted(path.name for path in shard_paths if path.stat().st_size == 0)
    if empty:
        raise SystemExit(f"{split}: empty or interrupted feature shards: {empty[:5]} (total={len(empty)})")
    summary[split] = {"total": expected_count, "datasets": dict(sorted(counts.items()))}

print(json.dumps({"preflight": "passed", "frozen_cache": summary}, ensure_ascii=False))
PY

echo "[preflight] passed; no feature extraction command is present in this continuation"
if ((CHECK_ONLY)); then
  echo "[check-only] complete; training was not started"
  exit 0
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>"$RESULT_ROOT/.run.lock"
  if ! flock -n 9; then
    echo "Another independent full run holds $RESULT_ROOT/.run.lock" >&2
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

command=(
  "$PYTHON_BIN" run_independent_datasets.py
  --device "$DEVICE"
  --result-root "$RESULT_ROOT"
  --data-root "$DATA_ROOT"
  --full-manifest-root "$FULL_MANIFEST_ROOT"
  --controlled-manifest-root "$CONTROLLED_MANIFEST_ROOT"
  --full-experiment-root "$FULL_EXPERIMENT_ROOT"
  --controlled-experiment-root "$CONTROLLED_EXPERIMENT_ROOT"
  --reference-checkpoint "$REFERENCE_CHECKPOINT"
  --train-cap 0
  --stage-a-epochs "$STAGE_A_EPOCHS"
  --stage-b-epochs "$STAGE_B_EPOCHS"
  --stage-c-epochs "$STAGE_C_EPOCHS"
)
if ((RESUME)); then
  command+=(--resume)
fi

echo "[train] result root: $RESULT_ROOT"
"${command[@]}"
echo "[complete] $(date --iso-8601=seconds)"
echo "[summary] $RESULT_ROOT/main_generality.json"
