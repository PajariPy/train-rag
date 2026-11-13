#!/usr/bin/env bash
# Show every run's state and copy finished runs' results to ../runs/NAME (logs and
# result.json; the model itself only with MODEL=1, it is ~600 MB).
#
# Usage: [MODEL=1] ./fetch.sh
set -uo pipefail
. "$(dirname "$0")/common.sh"
connect

for NAME in $($SSH 'ls /workspace/job/runs 2>/dev/null'); do
  REMOTE=/workspace/job/runs/$NAME
  CODE=$($SSH "cat $REMOTE/exit_code 2>/dev/null")

  if [ -z "$CODE" ]; then
    LAST=$($SSH "grep -E 'before \(|eval_dev_subset_mrr|trained in|after:|Error' $REMOTE/train.log $REMOTE/stdout.log 2>/dev/null | tail -1" | cut -c1-150)
    log "$NAME: running or queued. ${LAST:-waiting for the GPU}"
    continue
  fi

  mkdir -p "$ROOT/runs/$NAME"
  $SCP "root@$HOST:$REMOTE/train.log" "root@$HOST:$REMOTE/stdout.log" "$ROOT/runs/$NAME/" 2>/dev/null
  $SCP "root@$HOST:$REMOTE/result.json" "$ROOT/runs/$NAME/" 2>/dev/null
  [ "${MODEL:-0}" = 1 ] && $SCP -r "root@$HOST:$REMOTE/final" "$ROOT/runs/$NAME/" 2>/dev/null
  log "$NAME: finished (exit $CODE), copied to runs/$NAME"
done

. "$STATE"
HOURS=$(python3 -c "from datetime import datetime,timezone as t; print(round((datetime.now(t.utc)-datetime.fromisoformat('$STARTED'.replace('Z','+00:00'))).total_seconds()/3600,2))")
log "machine up ${HOURS} h at \$$PRICE/h"
