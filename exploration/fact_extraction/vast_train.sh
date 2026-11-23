#!/usr/bin/env bash
# Train the fact-extraction LoRA on a rented Vast.ai GPU, copy the results back, destroy the GPU.
#
# Rents the cheapest reliable GPU of type $GPU, installs the same packages the local test used
# (uv, transformers from main, peft, accelerate, wandb; plus flash-linear-attention when it
# installs), uploads train.py and data/sft_*.jsonl, runs the training in the background with
# W&B logging, polls until it ends, copies runs/$NAME/{adapter,train.log} back and destroys the
# instance on exit, whatever happens. Watch it on W&B (rag-train/train-rag-facts) or through
# the instance's OPEN button (Jupyter) in the Vast.ai console.
#
# Usage: NAME=run-name [GPU=RTX_4090] [MAX_HOURS=5] [TRAIN_ARGS="--epochs 3"] ./vast_train.sh
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
NAME=${NAME:?set NAME}
GPU=${GPU:-RTX_4090}
MAX_HOURS=${MAX_HOURS:-5}
TEMPLATE=${TEMPLATE:-305ac3ffd3e42e0d9ad1f4ae14729ec2}  # "PyTorch (Vast)"
TRAIN_ARGS=${TRAIN_ARGS:-}
ENV_FILE=$HOME/projects/rag-implementation/train-rag/.env
OUT=$HERE/runs/$NAME
V="uvx vastai"
PKGS='--with "transformers @ git+https://github.com/huggingface/transformers" --with peft --with accelerate --with torch==2.15.0 --with torchvision --with pillow --with wandb'

log() { echo "$(date -u +%H:%M:%S) $*"; }
credit() { $V show user --raw 2>/dev/null | python3 -c "import json,sys; u=json.load(sys.stdin); print(f\"\${u.get('credit',0):.2f}\")"; }

log "credit before: $(credit)"
OFFERS=$($V search offers "gpu_name=$GPU num_gpus=1 reliability>0.98 rentable=true disk_space>=80 inet_down>300 cuda_vers>=13.2" -o dph --raw 2>/dev/null \
  | python3 -c "import json,sys
for o in json.load(sys.stdin)[:3]: print(o['id'], round(o['dph_total'],3), o.get('geolocation','?').replace(' ','_'))")
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
cleanup() {
  if [ -n "$SSH" ]; then
    log "copying results back to $OUT"
    mkdir -p "$OUT"
    $SCP -r "root@$HOST:/workspace/job/out/adapter" "$OUT/" 2>/dev/null
    $SCP "root@$HOST:/workspace/job/out/train.log" "root@$HOST:/workspace/job/setup.log" "$OUT/" 2>/dev/null
  fi
  destroy
  sleep 3
  log "live instances: $($V show instances --raw 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"
  log "credit after: $(credit)"
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
    --label "fact-lora-$NAME" --raw 2>/dev/null \
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

$SSH 'mkdir -p /workspace/job/out'
$SCP "$HERE/train.py" "$HERE/data/sft_train.jsonl" "$HERE/data/sft_eval.jsonl" "root@$HOST:/workspace/job/" || exit 1
# The W&B key travels on ssh's stdin into a private file, never on a command line.
# HF_TOKEN_READ (optional, a read-only token) avoids Hugging Face's slow anonymous downloads.
{ grep "^WANDB_API_KEY=" "$ENV_FILE"; grep "^HF_TOKEN_READ=" "$ENV_FILE" | sed 's/^HF_TOKEN_READ=/HF_TOKEN=/'; } \
  | sed 's/^/export /' | $SSH 'umask 077; cat > /workspace/job/.secrets'
$SSH "cat >> /workspace/job/.secrets" <<EOF
export WANDB_ENTITY=rag-train WANDB_PROJECT=train-rag-facts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=0
EOF

# A real Triton kernel: flash-linear-attention compiles its kernels at run time, which needs a C
# compiler (the slim runtime image has none, so gcc is installed first). If this fails, the
# training runs on the slow DeltaNet path instead of crashing.
$SSH 'cat > /workspace/job/check_triton.py' <<'CHECK'
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
$SSH "cd /workspace/job && (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc g++ > /dev/null) > setup.log 2>&1; \
  (curl -LsSf https://astral.sh/uv/install.sh | sh) >> setup.log 2>&1 \
  && export PATH=\$HOME/.local/bin:\$PATH \
  && (timeout 900 uv run --no-project --python 3.13 $PKGS --with flash-linear-attention python check_triton.py >> setup.log 2>&1 \
      && echo '--with flash-linear-attention' > fla_flag || echo '' > fla_flag)"
FLA=$($SSH 'cat /workspace/job/fla_flag')
log "setup done; fast DeltaNet kernels: ${FLA:-not available, using the slow path}"

$SSH "cd /workspace/job && nohup bash -c '. ./.secrets; export PATH=\$HOME/.local/bin:\$PATH; \
  uv run --no-project --python 3.13 $PKGS $FLA python train.py --data . --out out --name $NAME $TRAIN_ARGS > out/stdout.log 2>&1; \
  echo \$? > out/exit_code' > /dev/null 2>&1 < /dev/null &" < /dev/null
log "training started: W&B rag-train/train-rag-facts, run $NAME; Vast.ai console OPEN for Jupyter"

DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 120
  CODE=$($SSH 'cat /workspace/job/out/exit_code 2>/dev/null')
  LAST=$($SSH "grep -E \"'loss'|'eval_loss'|held-out|trained in|Error\" /workspace/job/out/train.log /workspace/job/out/stdout.log 2>/dev/null | tail -1" | cut -c1-160)
  log "${LAST:-starting...}"
  [ -n "$CODE" ] && { log "training exited with code $CODE"; break; }
done
[ -z "${CODE:-}" ] && log "time limit of $MAX_HOURS h reached; stopping"
$SSH 'tail -5 /workspace/job/out/stdout.log' 2>/dev/null
