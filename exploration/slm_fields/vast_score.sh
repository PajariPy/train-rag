#!/usr/bin/env bash
# Score a trained field LoRA on the held-out documents: rent one RTX 4090, merge the adapter
# into Qwen3.5-2B (merge_lora.py), write the fields for every held-out document with vLLM
# (generate_vllm.py), copy outputs.jsonl back to runs/$NAME/ and destroy the machine on every
# exit. Scoring itself is local and free: score_heldout.py.
#
# Reports to the progress page's workflow step "Score the held-out documents (vLLM)" when KEY
# is set. Every ssh call has a timeout; setup and generation run detached and are polled.
# About 1 hour on a 4090 (~$0.45). Ask the user before running.
#
# Usage: KEY="<workflow key>" NAME=fields-r16-e2 [SKIP="offer ids"] [MAX_HOURS=3] \
#        exploration/slm_fields/vast_score.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$HERE/../.." && pwd)
NAME=${NAME:?set NAME}; KEY=${KEY:-}; MAX_HOURS=${MAX_HOURS:-3}
OUT=$HERE/runs/$NAME
STEP="Score the held-out documents (vLLM)"
TEMPLATE=305ac3ffd3e42e0d9ad1f4ae14729ec2  # "PyTorch (Vast)"
VLLM=0.33.0
V="uvx vastai"
OPTS="-i $HOME/.ssh/vastai -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30"
# The training's packages, for the merge (the adapter was saved with them).
MERGE_PKGS='--with "transformers @ git+https://github.com/huggingface/transformers" --with peft --with accelerate --with torch==2.15.0 --with torchvision --with pillow'

log() { echo "$(date -u +%H:%M:%S) $*"; }
wf() { [ -z "$KEY" ] && return 0; (cd "$ROOT_DIR" && timeout 60 uv run train-rag workflow "$1" "$KEY" "${@:2}") > /dev/null 2>&1 || true; }
credit() { $V show user --raw 2>/dev/null | python3 -c "import json,sys; print(f\"{json.load(sys.stdin).get('credit',0):.2f}\")"; }
remote() { timeout "${T:-60}" $SSH -n "$@"; }

[ -f "$OUT/adapter/adapter_model.safetensors" ] || { log "no adapter in $OUT"; exit 1; }
mkdir -p "$OUT"
# Compact prompts: the instructions once, then the held-out documents (not 12.7k copies).
python3 - "$HERE/data/sft_heldout.jsonl" "$OUT/prompts.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]).read().split("\n") if l.strip()]
with open(sys.argv[2], "w") as out:
    out.write(json.dumps({"system": rows[0]["messages"][0]["content"]}) + "\n")
    for r in rows:
        out.write(json.dumps({"doc_id": r["doc_id"], "source": r["source"],
                              "text": r["messages"][1]["content"]}, ensure_ascii=False) + "\n")
print(len(rows), "held-out documents")
PY
TOTAL=$(( $(wc -l < "$OUT/prompts.jsonl") - 1 ))

log "credit before: $(credit)"
wf step "$STEP" running --note "renting a GPU; credit before: \$$(credit)"
OFFERS=$($V search offers "gpu_name=RTX_4090 num_gpus=1 reliability>0.99 rentable=true disk_space>=80 inet_down>500 inet_down_cost<0.005 cuda_vers>=13.2" -o dph --raw 2>/dev/null \
  | python3 -c "import json,sys
for o in [o for o in json.load(sys.stdin) if str(o['id']) not in '${SKIP:-}'.split()][:3]: print(o['id'], round(o['dph_total'],3))")
[ -z "$OFFERS" ] && { log "no offer"; wf step "$STEP" failed --note "no RTX 4090 offer"; exit 1; }

INSTANCE=""; SSH=""; FINISHED=""
cleanup() {
  if [ -n "$SSH" ]; then
    log "copying results back to $OUT"
    timeout 600 $SCP "root@$HOST:/workspace/score/outputs.jsonl" "root@$HOST:/workspace/score/generate.log" \
      "root@$HOST:/workspace/score/setup.log" "$OUT/" 2>/dev/null
  fi
  if [ -n "$INSTANCE" ]; then
    log "destroying $INSTANCE"
    timeout 120 $V destroy instance "$INSTANCE" -y > /dev/null 2>&1
    sleep 3
  fi
  LIVE=$($V show instances --raw 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')
  GOT=$(wc -l < "$OUT/outputs.jsonl" 2>/dev/null || echo 0)
  log "live instances: $LIVE; outputs: $GOT/$TOTAL; credit after: \$$(credit)"
  if [ -n "$FINISHED" ]; then
    wf step "$STEP" success --note "$GOT/$TOTAL documents; GPU destroyed (live: $LIVE); credit: \$$(credit); score with score_heldout.py"
  else
    wf step "$STEP" failed --note "stopped with $GOT/$TOTAL documents; GPU destroyed (live: $LIVE); see the terminal log"
  fi
}
trap cleanup EXIT

while read -r OFFER_ID PRICE <&3; do
  log "renting offer $OFFER_ID at \$$PRICE/h"
  INSTANCE=$($V create instance "$OFFER_ID" --template_hash "$TEMPLATE" --disk 80 \
    --label "fields-score-$NAME" --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
  [ -z "$INSTANCE" ] && { log "create failed; next offer"; continue; }
  STATUS=""
  for _ in $(seq 1 60); do
    STATUS=$($V show instance "$INSTANCE" --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status'))" 2>/dev/null)
    [ "$STATUS" = "running" ] && break
    sleep 10
  done
  URL=$($V ssh-url "$INSTANCE" 2>/dev/null | tail -1)
  HOSTPORT=${URL#ssh://root@}; HOST=${HOSTPORT%:*}; PORT=${HOSTPORT##*:}
  CANDIDATE="ssh $OPTS -p $PORT root@$HOST"
  OK=""
  if [ "$STATUS" = "running" ]; then
    for _ in $(seq 1 30); do timeout 30 $CANDIDATE -n true 2>/dev/null && { OK=1; break; }; sleep 10; done
  fi
  if [ -z "$OK" ]; then
    log "offer $OFFER_ID never answered ($STATUS); next"
    timeout 120 $V destroy instance "$INSTANCE" -y > /dev/null 2>&1; INSTANCE=""; continue
  fi
  SSH=$CANDIDATE; SCP="scp $OPTS -P $PORT"; break
done 3<<<"$OFFERS"
[ -z "$SSH" ] && { log "no host booted"; exit 1; }
log "ssh ok ($HOST:$PORT), \$$PRICE/h"
wf step "$STEP" running --note "RTX 4090 at \$$PRICE/h; uploading and installing"

remote 'mkdir -p /workspace/score'
timeout 900 $SCP -r "$OUT/adapter" "$HERE/merge_lora.py" "$HERE/generate_vllm.py" "$OUT/prompts.jsonl" \
  "root@$HOST:/workspace/score/" || { log "upload failed"; exit 1; }

timeout 60 $SSH 'cat > /workspace/score/job.sh' <<JOB  # not remote(): -n would drop the heredoc
cd /workspace/score
(curl -LsSf https://astral.sh/uv/install.sh | sh) > setup.log 2>&1
export PATH=\$HOME/.local/bin:\$PATH
uv run --no-project --python 3.13 $MERGE_PKGS python merge_lora.py adapter merged >> setup.log 2>&1 || { echo merge > failed; exit 1; }
uv run --no-project --python 3.12 --with vllm==$VLLM python generate_vllm.py merged prompts.jsonl outputs.jsonl > generate.log 2>&1
echo \$? > exit_code
JOB
T=45 remote "cd /workspace/score && setsid nohup bash job.sh > /dev/null 2>&1 < /dev/null &"

DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
CODE=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 60
  [ "$(remote 'cat /workspace/score/failed 2>/dev/null')" = "merge" ] && { log "merge failed:"; remote 'tail -20 /workspace/score/setup.log'; exit 1; }
  CODE=$(remote 'cat /workspace/score/exit_code 2>/dev/null')
  LINE=$(remote "grep -o 'progress docs [0-9]*/[0-9]*' /workspace/score/generate.log 2>/dev/null | tail -1")
  if [ -n "$LINE" ]; then
    DONE=$(echo "$LINE" | sed -E 's/progress docs ([0-9]+)\/.*/\1/')
    wf progress "$STEP" "$DONE" "$TOTAL" --unit "documents" --note "vLLM on RTX 4090 at \$$PRICE/h"
    log "documents $DONE/$TOTAL"
  else
    log "setting up: $(remote 'tail -1 /workspace/score/setup.log 2>/dev/null' | cut -c1-120)"
  fi
  [ -n "$CODE" ] && break
done
if [ "$CODE" = "0" ]; then FINISHED=1; log "generation finished"; else
  log "generation ended with code ${CODE:-none (time limit)}"; remote 'tail -25 /workspace/score/generate.log'; fi
