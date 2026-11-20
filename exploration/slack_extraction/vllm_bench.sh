#!/usr/bin/env bash
# Rent one RTX 4090 on Vast.ai, measure vLLM's speed for Qwen3.5-2B on 1,000 Slack threads
# (vllm_bench.py), copy the result back and destroy the machine. One shot: the machine is
# destroyed on every exit, success or failure, and the whole run has a hard time limit.
#
# Needs data/bench_prompts.json (1,000 threads + the extraction instructions).
# Costs about $0.25 (~40 min at ~$0.34/h). Ask the user before running.
#
# Usage: exploration/slack_extraction/vllm_bench.sh
set -uo pipefail
. "$(dirname "$0")/../embedder_finetune/vast/common.sh"
HERE=$(cd "$(dirname "$0")" && pwd)
LIMIT=${LIMIT:-3000}  # seconds for setup + bench before giving up
VLLM=0.33.0
SKIP=${SKIP:-}  # offer ids to leave out (hosts that failed to boot)
TEMPLATE=305ac3ffd3e42e0d9ad1f4ae14729ec2  # "PyTorch (Vast)"

log "credit before: $(credit)"
OFFERS=$($V search offers "gpu_name=RTX_4090 num_gpus=1 reliability>0.98 rentable=true disk_space>=60 inet_down>300 inet_down_cost<0.005 cuda_vers>=13.2 geolocation notin [CN]" -o dph --raw 2>/dev/null \
  | python3 -c "import json,sys
for o in [o for o in json.load(sys.stdin) if str(o['id']) not in '$SKIP'.split()][:3]: print(o['id'], round(o['dph_total'],3))")
[ -z "$OFFERS" ] && { log "no offer"; exit 1; }

INSTANCE=""
destroy() {
  [ -z "$INSTANCE" ] && return
  log "destroying $INSTANCE"
  timeout 120 $V destroy instance "$INSTANCE" -y >/dev/null 2>&1
  INSTANCE=""
  sleep 3
  log "live instances: $($V show instances --raw 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"
  log "credit after: $(credit)"
}
trap 'destroy' EXIT

# Up to 3 offers; a host that doesn't boot in 6 minutes or never answers ssh is dropped.
while read -r OFFER_ID PRICE <&3; do  # offers on fd 3, so nothing in the loop can eat them
  log "renting RTX 4090 offer $OFFER_ID at \$$PRICE/h"
  INSTANCE=$($V create instance "$OFFER_ID" --template_hash "$TEMPLATE" --disk 60 \
    --label "vllm-bench" --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
  [ -z "$INSTANCE" ] && { log "create failed; next offer"; continue; }
  log "instance $INSTANCE"
  STATUS=""
  for _ in $(seq 1 36); do
    STATUS=$($V show instance "$INSTANCE" --raw 2>/dev/null \
      | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status'))" 2>/dev/null)
    [ "$STATUS" = "running" ] && break
    sleep 10
  done
  if [ "$STATUS" != "running" ]; then log "never started ($STATUS); next offer"; destroy; continue; fi
  URL=$($V ssh-url "$INSTANCE" 2>/dev/null | tail -1)
  HOSTPORT=${URL#ssh://root@}; HOST=${HOSTPORT%:*}; PORT=${HOSTPORT##*:}
  SSH="ssh $OPTS -p $PORT root@$HOST"
  SCP="scp $OPTS -P $PORT"
  CONNECTED=""
  for _ in $(seq 1 30); do $SSH -n true 2>/dev/null && { CONNECTED=1; break; }; sleep 10; done  # -n: keep the offer list on stdin
  if [ -z "$CONNECTED" ]; then log "ssh never answered; next offer"; destroy; continue; fi
  break
done 3<<<"$OFFERS"
[ -z "$INSTANCE" ] && { log "no host booted; giving up"; exit 1; }
log "ssh ok ($HOST:$PORT)"

$SSH 'mkdir -p /workspace/bench'
$SCP "$HERE/vllm_bench.py" "$HERE/data/bench_prompts.json" "root@$HOST:/workspace/bench/" \
  || { log "upload failed"; exit 1; }
grep "^HF_TOKEN_READ=" "$ENV_FILE" | sed 's/^HF_TOKEN_READ=/export HF_TOKEN=/' \
  | $SSH 'umask 077; cat > /workspace/bench/.secrets'

log "installing vLLM $VLLM, then the bench (limit ${LIMIT}s)"
timeout "$LIMIT" $SSH "cd /workspace/bench && . ./.secrets && (curl -LsSf https://astral.sh/uv/install.sh | sh) > setup.log 2>&1 \
  && export PATH=\$HOME/.local/bin:\$PATH \
  && uv run --no-project --python 3.12 --with vllm==$VLLM python vllm_bench.py bench_prompts.json > bench.log 2>&1"
CODE=$?
$SCP "root@$HOST:/workspace/bench/bench_result.json" "$HERE/data/" 2>/dev/null \
  && log "result: $(cat "$HERE/data/bench_result.json")" \
  || { log "no result (exit $CODE); last log lines:"; $SSH 'tail -25 /workspace/bench/bench.log /workspace/bench/setup.log' 2>/dev/null; }
$SCP "root@$HOST:/workspace/bench/bench.log" "$HERE/data/bench.log" 2>/dev/null
