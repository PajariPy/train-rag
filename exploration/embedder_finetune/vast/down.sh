#!/usr/bin/env bash
# Copy every run's results back, then destroy the rented machine (billing stops).
#
# Usage: [MODEL=1] ./down.sh
set -uo pipefail
. "$(dirname "$0")/common.sh"
connect

# 1. Results first: nothing on the machine survives the destroy.
"$HERE/fetch.sh"

# 2. Destroy (-y: without it the CLI waits for a confirmation and the machine keeps billing).
log "destroying $INSTANCE"
timeout 120 $V destroy instance "$INSTANCE" -y > /dev/null 2>&1
rm -f "$STATE"
sleep 3
log "live instances: $($V show instances --raw 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"
log "credit after: $(credit)"
