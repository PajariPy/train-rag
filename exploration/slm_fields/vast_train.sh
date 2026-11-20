#!/usr/bin/env bash
# Train the field-extractor LoRA (all sources) on a rented Vast.ai GPU, copy the results back,
# destroy the GPU. Adapted from exploration/fact_extraction/vast_train.sh.
#
# Reports every step to the progress page's workflow panel when KEY is set (the key printed
# by `train-rag workflow start`): renting, a short smoke run (SMOKE_STEPS optimizer steps,
# 3 generations), the full training with its step count and loss, copy back and destroy.
#
# Rents the cheapest reliable GPU of type $GPU, installs the same packages the local test used
# (uv, transformers from main, peft, accelerate, wandb; plus flash-linear-attention when it
# installs), uploads train.py and data/sft_{train,eval}.jsonl, runs the training in the background with
# W&B logging, polls until it ends, copies runs/$NAME/{adapter,train.log} back and destroys the
# instance on exit, whatever happens. Watch it on W&B (rag-train/train-rag-fields) or through
# the instance's OPEN button (Jupyter) in the Vast.ai console.
#
# Usage: KEY="<workflow key>" NAME=run-name [GPU=RTX_4090] [MAX_HOURS=14] [SMOKE_STEPS=10] [SKIP="offer ids"]
#        [TRAIN_ARGS="--epochs 2"] ./vast_train.sh
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
NAME=${NAME:?set NAME}
GPU=${GPU:-RTX_4090}
MAX_HOURS=${MAX_HOURS:-14}
SMOKE_STEPS=${SMOKE_STEPS:-10}
KEY=${KEY:-}
ROOT_DIR=$(cd "$HERE/../.." && pwd)
RENT="Rent a GPU (Vast.ai)"
SMOKE="Smoke run (10 steps)"
FULL="Full training, 2 epochs (W&B)"
BACK="Copy back and destroy the GPU"
TEMPLATE=${TEMPLATE:-305ac3ffd3e42e0d9ad1f4ae14729ec2}  # "PyTorch (Vast)"
TRAIN_ARGS=${TRAIN_ARGS:-}
ENV_FILE=$HOME/projects/rag-implementation/train-rag/.env
OUT=$HERE/runs/$NAME
V="uvx vastai"
PKGS='--with "transformers @ git+https://github.com/huggingface/transformers" --with peft --with accelerate --with torch==2.15.0 --with torchvision --with pillow --with wandb'

log() { echo "$(date -u +%H:%M:%S) $*"; }
remote() { timeout "${T:-60}" $SSH -n "$@"; }  # one ssh call that can't hang
wf() {  # report to the progress page's workflow panel; never fails the job
  [ -z "$KEY" ] && return 0
  (cd "$ROOT_DIR" && timeout 60 uv run train-rag workflow "$1" "$KEY" "${@:2}") > /dev/null 2>&1 || true
}
credit() { $V show user --raw 2>/dev/null | python3 -c "import json,sys; u=json.load(sys.stdin); print(f\"\${u.get('credit',0):.2f}\")"; }

log "credit before: $(credit)"
CURRENT=$RENT; wf step "$RENT" running --note "credit before: $(credit)"
OFFERS=$($V search offers "gpu_name=$GPU num_gpus=1 reliability>0.99 rentable=true disk_space>=80 inet_down>500 inet_down_cost<0.005 cuda_vers>=13.2" -o dph --raw 2>/dev/null \
  | python3 -c "import json,sys
offers = [o for o in json.load(sys.stdin) if str(o['id']) not in '${SKIP:-}'.split()]
for o in offers[:3]: print(o['id'], round(o['dph_total'],3), o.get('geolocation','?').replace(' ','_'))")
[ -z "$OFFERS" ] && { log "no $GPU offer found"; exit 1; }
OPTS="-i $HOME/.ssh/vastai -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30"

INSTANCE=""
SSH=""
destroy() {  # destroy the current instance, if any
  [ -z "$INSTANCE" ] && return
  log "destroying $INSTANCE"
  timeout 120 $V destroy instance "$INSTANCE" -y >/dev/null 2>&1
  INSTANCE=""
}
CURRENT=""  # the workflow step running now, marked failed if the script dies in it
cleanup() {
  [ -n "$CURRENT" ] && wf step "$CURRENT" failed --note "the script stopped here; see the terminal log"
  wf step "$BACK" running
  if [ -n "$SSH" ]; then
    log "copying results back to $OUT"
    mkdir -p "$OUT"
    $SCP -r "root@$HOST:/workspace/job/out/adapter" "$OUT/" 2>/dev/null
    $SCP "root@$HOST:/workspace/job/smoke/train.log" "$OUT/smoke_train.log" 2>/dev/null
    $SCP "root@$HOST:/workspace/job/out/train.log" "root@$HOST:/workspace/job/setup.log" "$OUT/" 2>/dev/null
  fi
  destroy
  sleep 3
  log "live instances: $($V show instances --raw 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"
  log "credit after: $(credit)"
  wf step "$BACK" success --note "credit after: $(credit)"
}
trap cleanup EXIT

# Rent the cheapest offer that boots and accepts ssh; a host that doesn't is destroyed and the
# next offer tried (up to 3).
while read -r OFFER_ID PRICE WHERE; do
  log "renting $GPU offer $OFFER_ID at \$$PRICE/h in $WHERE"
  # Vast's own "PyTorch (Vast)" template: an image cached on many hosts (the plain
  # pytorch/pytorch image left hosts "loading" for 15+ minutes), with Jupyter and the Instance
  # Portal behind the OPEN button. Our packages, torch included, are installed by uv anyway.
  INSTANCE=$($V create instance "$OFFER_ID" --template_hash "$TEMPLATE" --disk 80 \
    --label "fields-lora-$NAME" --raw 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
  [ -z "$INSTANCE" ] && { log "create failed; next offer"; continue; }
  log "instance $INSTANCE"
  STATUS=""
  for _ in $(seq 1 90); do  # up to 15 minutes to boot (first image pull is slow)
    STATUS=$($V show instance "$INSTANCE" --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status'))" 2>/dev/null)
    [ "$STATUS" = "running" ] && break
    sleep 10
  done
  if [ "$STATUS" != "running" ]; then log "never started (status $STATUS); next offer"; destroy; continue; fi
  URL=$($V ssh-url "$INSTANCE" 2>/dev/null | tail -1)
  HOSTPORT=${URL#ssh://root@}; HOST=${HOSTPORT%:*}; PORT=${HOSTPORT##*:}
  CANDIDATE="ssh $OPTS -p $PORT root@$HOST"
  CONNECTED=""
  for _ in $(seq 1 30); do $CANDIDATE true 2>/dev/null && { CONNECTED=1; break; }; sleep 10; done  # 5 minutes
  if [ -z "$CONNECTED" ]; then log "ssh never answered on $HOST:$PORT; next offer"; destroy; continue; fi
  SSH=$CANDIDATE
  SCP="scp $OPTS -P $PORT"
  break
done <<<"$OFFERS"
[ -z "$SSH" ] && { log "no host accepted ssh; giving up"; exit 1; }
log "running, ssh ok ($HOST:$PORT); setting up"
wf step "$RENT" running --note "$GPU at \$$PRICE/h in $WHERE; installing packages"

timeout 60 $SSH 'mkdir -p /workspace/job/out'
timeout 1200 $SCP "$HERE/train.py" "$HERE/data/sft_train.jsonl" "$HERE/data/sft_eval.jsonl" "root@$HOST:/workspace/job/" || exit 1
# The W&B key travels on ssh's stdin into a private file, never on a command line.
# HF_TOKEN_READ (optional, a read-only token) avoids Hugging Face's slow anonymous downloads.
{ grep "^WANDB_API_KEY=" "$ENV_FILE"; grep "^HF_TOKEN_READ=" "$ENV_FILE" | sed 's/^HF_TOKEN_READ=/HF_TOKEN=/'; } \
  | sed 's/^/export /' | timeout 60 $SSH 'umask 077; cat > /workspace/job/.secrets'
timeout 60 $SSH "cat >> /workspace/job/.secrets" <<EOF
export WANDB_ENTITY=rag-train WANDB_PROJECT=train-rag-fields PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=0
EOF

# A real Triton kernel: flash-linear-attention compiles its kernels at run time, which needs a C
# compiler (the slim runtime image has none, so gcc is installed first). If this fails, the
# training runs on the slow DeltaNet path instead of crashing.
timeout 60 $SSH 'cat > /workspace/job/check_triton.py' <<'CHECK'
import torch
import triton
import triton.language as tl

import fla  # noqa: F401


@triton.jit
def add_one(x, y):
    tl.store(y, tl.load(x) + 1)


x = torch.ones(1, device="cuda")
y = torch.empty_like(x)
add_one[(1,)](x, y)
print("triton kernel ok", y.item())
CHECK
# Setup runs detached on the host and is polled: a dropped ssh connection (it happened on
# 20 Nov) must not cut it short or make its result look empty.
timeout 60 $SSH 'cat > /workspace/job/setup.sh' <<SETUP
cd /workspace/job
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc g++ > /dev/null) > setup.log 2>&1
(curl -LsSf https://astral.sh/uv/install.sh | sh) >> setup.log 2>&1
export PATH=\$HOME/.local/bin:\$PATH
if timeout 900 uv run --no-project --python 3.13 $PKGS --with flash-linear-attention python check_triton.py >> setup.log 2>&1; then
  echo '--with flash-linear-attention' > fla_flag
else
  echo none > fla_flag
fi
touch setup_done
SETUP
T=45 remote "cd /workspace/job && rm -f fla_flag setup_done && setsid nohup bash setup.sh > /dev/null 2>&1 < /dev/null &"
FLA=""
for _ in $(seq 1 60); do  # up to 30 minutes
  sleep 30
  [ "$(remote 'test -f /workspace/job/setup_done && echo yes')" = "yes" ] || continue
  FLA=$(remote 'cat /workspace/job/fla_flag'); break
done
if [ -z "$FLA" ] || [ "$FLA" = "none" ]; then
  log "fast DeltaNet kernels not available (${FLA:-setup never finished}); not starting a ~12x slower run"
  remote 'tail -15 /workspace/job/setup.log'
  exit 1
fi
log "setup done; fast DeltaNet kernels: $FLA"

wf step "$RENT" success --note "$GPU at \$$PRICE/h in $WHERE; fast kernels: ${FLA:-no}"

run_train() {  # $1 = out dir, rest = train.py args; runs in the background on the GPU
  local out=$1; shift
  T=45 remote "cd /workspace/job && mkdir -p $out && rm -f $out/exit_code && setsid nohup bash -c '. ./.secrets; export PATH=\$HOME/.local/bin:\$PATH; \
    uv run --no-project --python 3.13 $PKGS $FLA python train.py --data . --out $out $* > $out/stdout.log 2>&1; \
    echo \$? > $out/exit_code' > /dev/null 2>&1 < /dev/null &"
  # The launch call may time out on a flaky connection even when the job started: check.
  for _ in 1 2 3 4 5 6; do
    [ "$(remote "test -f /workspace/job/$out/stdout.log && echo yes")" = "yes" ] && return 0
    sleep 20
  done
  log "the $out job never started"; exit 1
}
wait_train() {  # $1 = out dir, $2 = workflow step, $3 = deadline (epoch s); sets CODE
  CODE=""
  while [ "$(date +%s)" -lt "$3" ]; do
    sleep 60
    CODE=$(remote "cat /workspace/job/$1/exit_code 2>/dev/null")
    STEP=$(remote "grep -o 'progress step [0-9]*/[0-9]* loss [-0-9.]*' /workspace/job/$1/train.log 2>/dev/null | tail -1")
    if [ -n "$STEP" ]; then
      DONE=$(echo "$STEP" | sed -E 's/progress step ([0-9]+)\/([0-9]+).*/\1/')
      TOTAL=$(echo "$STEP" | sed -E 's/progress step ([0-9]+)\/([0-9]+).*/\2/')
      LOSS=${STEP##* }
      wf progress "$2" "$DONE" "$TOTAL" --unit "training steps" --note "loss $LOSS · W&B rag-train/train-rag-fields, run $NAME"
      log "$2: step $DONE/$TOTAL, loss $LOSS"
    else
      log "$2: $(remote "tail -1 /workspace/job/$1/stdout.log 2>/dev/null" | cut -c1-140)"
    fi
    [ -n "$CODE" ] && break
  done
}

# Smoke run: a few optimizer steps and 3 generations, to catch setup errors cheaply.
CURRENT=$SMOKE; wf step "$SMOKE" running --note "$SMOKE_STEPS steps, 3 generations"
run_train smoke --name "$NAME-smoke" --max-steps "$SMOKE_STEPS" --generate 2
wait_train smoke "$SMOKE" $(( $(date +%s) + 3600 ))
if [ "${CODE:-}" != "0" ]; then
  log "smoke run failed (exit ${CODE:-none}); last lines:"; remote 'tail -20 /workspace/job/smoke/stdout.log' 2>/dev/null
  exit 1
fi
CHECK=$(remote "grep 'held-out check' /workspace/job/smoke/train.log | tail -1" | sed -E "s/.*held-out check: //" | cut -c1-200)
log "smoke ok: $CHECK"
wf step "$SMOKE" success --note "$CHECK"

# The full training.
CURRENT=$FULL; wf step "$FULL" running --note "W&B rag-train/train-rag-fields, run $NAME"
run_train out --name "$NAME" $TRAIN_ARGS
log "training started: W&B rag-train/train-rag-fields, run $NAME; Vast.ai console OPEN for Jupyter"
wait_train out "$FULL" $(( $(date +%s) + MAX_HOURS * 3600 ))
if [ "${CODE:-}" = "0" ]; then
  CHECK=$(remote "grep 'held-out check' /workspace/job/out/train.log | tail -1" | sed -E "s/.*held-out check: //" | cut -c1-200)
  wf step "$FULL" success --note "$CHECK"
  CURRENT=""
else
  log "training ended with code ${CODE:-none (time limit)}"
fi
remote 'tail -5 /workspace/job/out/stdout.log' 2>/dev/null
