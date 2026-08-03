#!/usr/bin/env bash
# Gap-fill collection runner for one agent family (claude|codex).
# Each round, per model: compute farm tasks with no completed clean trial,
# atomically materialize a fill dir, and run Harbor over it. State is always
# recomputed from disk, including suffixed jobs dirs from prior waves.
cd "$(dirname "$0")"
set -euo pipefail

PY=.venv/bin/python
FAMILY=${1:-}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
CONCURRENCY=${CONCURRENCY:-2}
WAIT_FOR=${WAIT_FOR:-}

case "$FAMILY" in
claude)
  AGENT=claude-code
  DEFAULT_MODELS="claude-opus-5 claude-sonnet-4-6 claude-haiku-4-5"
  ;;
codex)
  AGENT=codex
  DEFAULT_MODELS="gpt-5.6-sol gpt-5.6-terra gpt-5.6-luna"
  ;;
*)
  echo "usage: $0 {claude|codex}" >&2
  exit 2
  ;;
esac

MODELS=${MODELS:-$DEFAULT_MODELS}
mkdir -p logs harbor_runs

while :; do
  any_missing=0
  for m in $MODELS; do
    fill="harbor_tasks/fill-$m"
    $PY fill_missing.py "$m" --max-attempts "$MAX_ATTEMPTS" \
      --materialize "$fill" > "logs/missing_$m.txt"
    n=$(wc -l < "logs/missing_$m.txt" | tr -d ' ')
    [ "$n" -eq 0 ] && continue
    any_missing=1
    echo "=== $AGENT $m round: $n tasks $(date)"
    if [ "$FAMILY" = claude ]; then
      # Re-read per round: the OAuth access token can expire mid-farm.
      TOK=$(security find-generic-password -s 'Claude Code-credentials' -w \
            | $PY -c "import json,sys;print(json.load(sys.stdin)['claudeAiOauth']['accessToken'])")
      AE=(--ae "CLAUDE_CODE_OAUTH_TOKEN=$TOK" --ae "CLAUDE_FORCE_OAUTH=1")
    else
      AE=(--ae "CODEX_FORCE_AUTH_JSON=1")
    fi
    .venv/bin/harbor run -p "$fill" -a "$AGENT" -m "$m" -k 1 \
      -n "$CONCURRENCY" \
      --jobs-dir "harbor_runs/farm-$m" "${AE[@]}"
  done
  if [ "$any_missing" -eq 0 ]; then
    if [ -z "$WAIT_FOR" ] || [ -e "$WAIT_FOR" ]; then
      break
    fi
    echo "=== family $FAMILY idle; waiting for $WAIT_FOR $(date)"
    sleep 300
  fi
done
echo "=== family $FAMILY complete $(date)"
