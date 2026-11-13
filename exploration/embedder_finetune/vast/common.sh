# Shared settings for the embedder fine-tune scripts on Vast.ai (sourced, not run).
#
# The rented machine's id, host and port live in $STATE, written by up.sh and removed by
# down.sh, so every script talks to the same machine.

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)                 # exploration/embedder_finetune
# id, host, port of the rented machine; MACHINE=name keeps a second rental apart
STATE=$HERE/.instance${MACHINE:+-$MACHINE}
ENV_FILE=$HOME/projects/rag-implementation/train-rag/.env
V="uvx vastai"
OPTS="-i $HOME/.ssh/vastai -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30"
# The same versions as the laptop run (uv.lock).
PKGS='--with torch==2.15.0 --with transformers==5.22.1 --with sentence-transformers==6.2.0
  --with datasets==5.2.0 --with accelerate==1.15.0 --with wandb==0.31.0'

log() { echo "$(date -u +%H:%M:%S) $*"; }

credit() {
  $V show user --raw 2>/dev/null \
    | python3 -c "import json,sys; print(f\"\${json.load(sys.stdin).get('credit',0):.2f}\")"
}

connect() {  # read the state file and set SSH, SCP, HOST
  [ -f "$STATE" ] || { log "no machine rented (run up.sh first)"; exit 1; }
  # shellcheck disable=SC1090
  . "$STATE"
  SSH="ssh $OPTS -p $PORT root@$HOST"
  SCP="scp $OPTS -P $PORT"
}
