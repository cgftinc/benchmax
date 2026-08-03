#!/usr/bin/env bash
# Sequential gate chain; after each repo passes, its gated tasks join the
# farm dir so the collection runners pick them up next round.
cd "$(dirname "$0")"
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 REPO [REPO ...]" >&2
  exit 2
fi

K=${K:-3}
CONCURRENCY=${CONCURRENCY:-4}
FARM=${FARM:-harbor_tasks/farm}
JOBS_ROOT=${JOBS_ROOT:-harbor_runs}
mkdir -p logs "$FARM" "$JOBS_ROOT"

for r in "$@"; do
  source_dir="harbor_tasks/$r"
  if [ ! -d "$source_dir" ]; then
    echo "missing task directory: $source_dir" >&2
    exit 2
  fi
  echo "=== gating $r $(date)"
  .venv/bin/python gate_tasks.py "$source_dir" -k "$K" -n "$CONCURRENCY" \
    --jobs-dir "$JOBS_ROOT/gate-$r" --promote-to "$FARM" \
    2>&1 | tee -a "logs/gate_$r.log"
done
echo "=== all gates done $(date)"
