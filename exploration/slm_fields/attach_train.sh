#!/usr/bin/env bash
# Run the full field-LoRA training on a Vast.ai machine that vast_train.sh already rented and
# set up, watch it, copy the results back and destroy the machine (on every exit).
#
# Written on 20 Nov after vast_train.sh lost its ssh connection during setup: it read the
# fast-kernel flag as empty (the slow path, ~12x slower) and its launch call never returned.
# Here every ssh call has a timeout, the job is launched fully detached (setsid), and the
# flag is read with retries.
#
# Usage: INSTANCE=<id> HOST=<ip> PORT=<port> KEY="<workflow key>" NAME=<run name> \
#        [MAX_HOURS=14] [TRAIN_ARGS=...] exploration/slm_fields/attach_train.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$HERE/../.." && pwd)
INSTANCE=${INSTANCE:?}; HOST=${HOST:?}; PORT=${PORT:?}; KEY=${KEY:-}; NAME=${NAME:?}
MAX_HOURS=${MAX_HOURS:-14}
TRAIN_ARGS=${TRAIN_ARGS:-}
OUT=$HERE/runs/$NAME
FULL="Full training, 2 epochs (W&B)"
BACK="Copy back and destroy the GPU"
V="uvx vastai"
OPTS="-i $HOME/.ssh/vastai -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30"
SSH="ssh $OPTS -p $PORT root@$HOST"
SCP="scp $OPTS -P $PORT"
PKGS='--with "transformers @ git+https://github.com/huggingface/transformers" --with peft --with accelerate --with torch==2.15.0 --with torchvision --with pillow --with wandb'

log() { echo "$(date -u +%H:%M:%S) $*"; }
wf() {  # report to the progress page's workflow panel; never fails the job
  [ -z "$KEY" ] && return 0
  (cd "$ROOT_DIR" && timeout 60 uv run train-rag workflow "$1" "$KEY" "${@:2}") > /dev/null 2>&1 || true
}
remote() { timeout "${T:-60}" $SSH -n "$@"; }  # one ssh call that can't hang
credit() { $V show user --raw 2>/dev/null | python3 -c "import json,sys; print(f\"{json.load(sys.stdin).get('credit',0):.2f}\")"; }

CURRENT=""
cleanup() {
  [ -n "$CURRENT" ] && wf step "$CURRENT" failed --note "the script stopped here; see the terminal log"
  wf step "$BACK" running
  log "copying results back to $OUT"
  mkdir -p "$OUT"
  timeout 600 $SCP -r "root@$HOST:/workspace/job/out/adapter" "$OUT/" 2>/dev/null
  timeout 120 $SCP "root@$HOST:/workspace/job/out/train.log" "root@$HOST:/workspace/job/out/stdout.log" \
    "root@$HOST:/workspace/job/setup.log" "$OUT/" 2>/dev/null
  timeout 120 $SCP "root@$HOST:/workspace/job/smoke/train.log" "$OUT/smoke_train.log" 2>/dev/null
  log "destroying $INSTANCE"
  timeout 120 $V destroy instance "$INSTANCE" -y > /dev/null 2>&1
  sleep 3
  LIVE=$($V show instances --raw 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')
  log "live instances: $LIVE; credit after: \$$(credit)"
  wf step "$BACK" success --note "adapter in exploration/slm_fields/runs/$NAME; live instances: $LIVE; credit: \$$(credit)"
}
trap cleanup EXIT

FLA=""
for _ in 1 2 3 4 5; do FLA=$(remote 'cat /workspace/job/fla_flag' 2>/dev/null) && [ -n "$FLA" ] && break; sleep 10; done
[ -z "$FLA" ] && { log "fast kernels not confirmed; not starting a 50-hour slow run"; exit 1; }
log "fast kernels: $FLA"

remote "pkill -f 'train.py --data . --out smoke' || true"  # the smoke run's sample generations
sleep 5
CURRENT=$FULL; wf step "$FULL" running --note "W&B rag-train/train-rag-fields, run $NAME"
remote "cd /workspace/job && mkdir -p out && rm -f out/exit_code && setsid nohup bash -c '. ./.secrets; export PATH=\$HOME/.local/bin:\$PATH; \
  uv run --no-project --python 3.13 $PKGS $FLA python train.py --data . --out out --name $NAME $TRAIN_ARGS > out/stdout.log 2>&1; \
  echo \$? > out/exit_code' > /dev/null 2>&1 < /dev/null &" || { log "launch failed"; exit 1; }
log "training started: W&B rag-train/train-rag-fields, run $NAME"

DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
CODE=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 60
  CODE=$(remote 'cat /workspace/job/out/exit_code 2>/dev/null')
  STEP=$(remote "grep -o 'progress step [0-9]*/[0-9]* loss [-0-9.]*' /workspace/job/out/train.log 2>/dev/null | tail -1")
  if [ -n "$STEP" ]; then
    DONE=$(echo "$STEP" | sed -E 's/progress step ([0-9]+)\/([0-9]+).*/\1/')
    TOTAL=$(echo "$STEP" | sed -E 's/progress step ([0-9]+)\/([0-9]+).*/\2/')
    LOSS=${STEP##* }
    wf progress "$FULL" "$DONE" "$TOTAL" --unit "training steps" --note "loss $LOSS · W&B rag-train/train-rag-fields, run $NAME"
    log "step $DONE/$TOTAL, loss $LOSS"
  else
    log "$(remote 'tail -1 /workspace/job/out/stdout.log 2>/dev/null' | cut -c1-140)"
  fi
  [ -n "$CODE" ] && break
done

if [ "$CODE" = "0" ]; then
  CHECK=$(remote "grep 'held-out check' /workspace/job/out/train.log | tail -1" | sed -E "s/.*held-out check: //" | cut -c1-200)
  wf step "$FULL" success --note "$CHECK"
  CURRENT=""
  log "training finished: $CHECK"
else
  log "training ended with code ${CODE:-none (time limit)}"
  remote 'tail -20 /workspace/job/out/stdout.log'
fi
