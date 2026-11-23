#!/usr/bin/env bash
# Start one fine-tune run on the rented machine and return at once.
#
# Runs wait for each other (one GPU, a lock file), so several can be queued back to back.
# train.py is uploaded again first, so an edit on the laptop is used by the next run.
# Each run writes to /workspace/job/runs/NAME and logs to W&B (rag-train/train-rag-embedder).
#
# Usage: ./run.sh NAME [train.py arguments...]
#   ./run.sh lr1e-5-s20 --lr 1e-5 --max-steps 20 --mini-batch 16
set -uo pipefail
. "$(dirname "$0")/common.sh"
connect

NAME=${1:?usage: run.sh NAME [train.py arguments...]}
shift
ARGS="$*"

# 1. Fresh copy of the training script.
$SCP "$ROOT/train.py" "root@$HOST:/workspace/job/train.py" || exit 1

# 2. Write the job as a script on the machine, then start it fully detached (setsid), so
#    this ssh returns at once. The lock queues it behind any run still holding the GPU.
$SSH "mkdir -p /workspace/job/runs/$NAME && cat > /workspace/job/runs/$NAME/job.sh" <<JOB
cd /workspace/job
. ./.secrets
export PATH=\$HOME/.local/bin:\$PATH
flock gpu.lock uv run --no-project --python 3.13 $(echo $PKGS) \\
  python train.py --data data --out runs/$NAME --name $NAME $ARGS > runs/$NAME/stdout.log 2>&1
echo \$? > runs/$NAME/exit_code
JOB
$SSH -n "setsid nohup bash /workspace/job/runs/$NAME/job.sh > /dev/null 2>&1 < /dev/null &"

log "queued $NAME ($ARGS); check with fetch.sh"
