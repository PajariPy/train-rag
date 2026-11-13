#!/usr/bin/env bash
# Rent one GPU on Vast.ai and set it up for several embedder fine-tune runs. It stays up
# until down.sh (it bills by the hour, so always end with down.sh).
#
# 1. Rent the cheapest reliable $GPU that boots and answers ssh (up to 3 offers tried).
# 2. Install uv and the same package versions as the laptop; download gte-modernbert-base.
# 3. Upload train.py and the training data (training set, dev set, document texts, ~190 MB).
# 4. Write the machine's id, host and port to vast/.instance for run.sh, fetch.sh, down.sh.
#
# If setup fails, the machine is destroyed right away.
#
# Usage: [GPU=RTX_4090] [NUM_GPUS=1] [MACHINE=name] [FILTER='geolocation notin [CN]'] ./up.sh
set -uo pipefail
. "$(dirname "$0")/common.sh"

GPU=${GPU:-RTX_4090}
NUM_GPUS=${NUM_GPUS:-1}
FILTER=${FILTER:-}  # extra offer filter, e.g. hosts outside China (Hugging Face is blocked there)
TEMPLATE=${TEMPLATE:-305ac3ffd3e42e0d9ad1f4ae14729ec2}  # "PyTorch (Vast)", cached on many hosts
[ -f "$STATE" ] && { log "a machine is already rented ($STATE); run down.sh first"; exit 1; }

log "credit before: $(credit)"
OFFERS=$($V search offers "gpu_name=$GPU num_gpus=$NUM_GPUS reliability>0.98 rentable=true disk_space>=60 inet_down>300 cuda_vers>=13.2 $FILTER" -o dph --raw 2>/dev/null \
  | python3 -c "import json,sys
for o in json.load(sys.stdin)[:3]: print(o['id'], round(o['dph_total'],3), o.get('geolocation','?').replace(' ','_'))")
[ -z "$OFFERS" ] && { log "no $GPU offer found"; exit 1; }

INSTANCE=""
destroy() {
  [ -z "$INSTANCE" ] && return
  log "destroying $INSTANCE"
  timeout 120 $V destroy instance "$INSTANCE" -y >/dev/null 2>&1
  INSTANCE=""
}
trap 'destroy' EXIT  # cleared once setup succeeds

# 1. Rent.
while read -r OFFER_ID PRICE WHERE; do
  log "renting ${NUM_GPUS}x $GPU offer $OFFER_ID at \$$PRICE/h in $WHERE"
  INSTANCE=$($V create instance "$OFFER_ID" --template_hash "$TEMPLATE" --disk 60 \
    --label "embedder-finetune" --raw 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
  [ -z "$INSTANCE" ] && { log "create failed; next offer"; continue; }
  log "instance $INSTANCE"

  STATUS=""
  for _ in $(seq 1 90); do  # up to 15 minutes to boot
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
  for _ in $(seq 1 30); do $SSH true 2>/dev/null && { CONNECTED=1; break; }; sleep 10; done
  if [ -z "$CONNECTED" ]; then log "ssh never answered; next offer"; destroy; continue; fi
  break
done <<<"$OFFERS"
[ -z "$INSTANCE" ] && { log "no host accepted ssh; giving up"; exit 1; }
log "running, ssh ok ($HOST:$PORT, \$$PRICE/h)"

# 2. Secrets and packages. The W&B key travels on ssh's stdin into a private file.
$SSH 'mkdir -p /workspace/job/runs'
{ grep "^WANDB_API_KEY=" "$ENV_FILE"; grep "^HF_TOKEN_READ=" "$ENV_FILE" | sed 's/^HF_TOKEN_READ=/HF_TOKEN=/'; } \
  | sed 's/^/export /' | $SSH 'umask 077; cat > /workspace/job/.secrets'
echo "export WANDB_ENTITY=rag-train WANDB_PROJECT=train-rag-embedder PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  | $SSH 'cat >> /workspace/job/.secrets'

log "installing packages and downloading the model (a few minutes)"
$SSH "cd /workspace/job && (curl -LsSf https://astral.sh/uv/install.sh | sh) > setup.log 2>&1 \
  && export PATH=\$HOME/.local/bin:\$PATH \
  && uv run --no-project --python 3.13 $(echo $PKGS) python -c \
     'from sentence_transformers import SentenceTransformer; import torch; \
      SentenceTransformer(\"Alibaba-NLP/gte-modernbert-base\"); print(\"ready\", torch.cuda.get_device_name())' \
     >> setup.log 2>&1" || { log "setup failed:"; $SSH 'tail -20 /workspace/job/setup.log'; exit 1; }
log "$($SSH 'tail -1 /workspace/job/setup.log')"

# 3. Code and data.
$SSH -n "mkdir -p /workspace/job/data"
$SCP "$ROOT/train.py" "root@$HOST:/workspace/job/" || { log "upload failed"; exit 1; }
$SCP "$ROOT/data/training_set.jsonl" "$ROOT/data/dev_set.jsonl" "$ROOT/data/documents.jsonl" \
  "root@$HOST:/workspace/job/data/" || { log "upload failed"; exit 1; }

# 4. Remember the machine; from here on only down.sh destroys it.
printf 'INSTANCE=%s\nHOST=%s\nPORT=%s\nPRICE=%s\nSTARTED=%s\n' \
  "$INSTANCE" "$HOST" "$PORT" "$PRICE" "$(date -u +%FT%TZ)" > "$STATE"
trap - EXIT
log "ready: instance $INSTANCE at \$$PRICE/h. Start runs with run.sh; end with down.sh."
