#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

feature_pid="${1:?feature extraction pid is required}"
while kill -0 "$feature_pid" 2>/dev/null; do
  sleep 30
done

exec "$PROJECT_ROOT/scripts/continue_visual_jev_independent_full.sh"
