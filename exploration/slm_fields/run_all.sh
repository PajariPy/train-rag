#!/usr/bin/env bash
# The SLM fine-tuning job, end to end, reported step by step to the progress page:
#   1. label the 2,000 extra documents with DeepSeek (waits for off-peak prices first);
#   2. build the training set (build_sft.py);
#   3-6. rent a GPU, smoke run, full training, copy back and destroy (vast_train.sh).
# Scoring on the held-out documents is a separate step, run after this.
#
# Usage, from the repo root: KEY="<workflow key>" NAME=fields-r16-e2 exploration/slm_fields/run_all.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
KEY=${KEY:?set KEY to the workflow key}
NAME=${NAME:-fields-r16-e2}
LABEL="Label 2,000 extra documents (DeepSeek)"
BUILD="Build the training set"
cd "$ROOT"
log() { echo "$(date -u +%H:%M:%S) $*"; }
wf() { timeout 60 uv run train-rag workflow "$1" "$KEY" "${@:2}" > /dev/null 2>&1 || true; }

# DeepSeek doubles its prices at peak hours (UTC 01-03 and 06-09 on weekdays): wait them out.
until uv run python -c "import sys; sys.path.insert(0, 'exploration/slack_extraction'); \
from label import peak_now; sys.exit(1 if peak_now() else 0)" 2>/dev/null; do
  wf step "$LABEL" queued --note "waiting for DeepSeek off-peak prices"
  log "peak hours; waiting"
  sleep 120
done

log "labelling the extra batch"
if ! uv run python exploration/slack_extraction/label.py --use train --batch extra-2026-11-20 \
    --workflow "$KEY" --step "$LABEL"; then
  log "labelling failed"; exit 1
fi

wf step "$BUILD" running
if ! OUTPUT=$(uv run python exploration/slm_fields/build_sft.py 2>&1); then
  echo "$OUTPUT"; wf step "$BUILD" failed --note "build_sft.py failed"; exit 1
fi
echo "$OUTPUT"
TRAIN=$(echo "$OUTPUT" | grep -o 'sft_train.jsonl: [0-9,]*')
wf step "$BUILD" success --note "$TRAIN examples"

KEY="$KEY" NAME="$NAME" "$HERE/vast_train.sh"
