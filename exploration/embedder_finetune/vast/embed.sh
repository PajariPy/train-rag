#!/usr/bin/env bash
# Embed the whole corpus with a fine-tuned model on a rented multi-GPU machine.
#
# Needs a machine from `MACHINE=embed NUM_GPUS=4 ./up.sh` and data/corpus.jsonl.gz from
# export_corpus.py. Then:
# 1. upload the corpus, the model folder and embed_corpus.py (each only once);
# 2. start one embed_corpus.py per GPU, each on its own shard of the corpus;
# 3. wait until every shard has written its done marker;
# 4. copy the vector parts back to data/vectors/ (for import_vectors.py).
# A rerun skips parts already embedded (the copy-back is repeated).
#
# Usage: MACHINE=embed ./embed.sh MODEL_DIR
set -uo pipefail
. "$(dirname "$0")/common.sh"
connect

MODEL_DIR=${1:?usage: embed.sh MODEL_DIR}
SHARDS=$($SSH -n 'nvidia-smi -L | wc -l')
OUT=$ROOT/data/vectors

# 1. Corpus, model and script (corpus and model only once).
$SSH -n 'mkdir -p /workspace/job/data /workspace/job/model /workspace/job/vectors'
if ! $SSH -n 'test -f /workspace/job/data/corpus.jsonl.gz'; then
  $SCP "$ROOT/data/corpus.jsonl.gz" "root@$HOST:/workspace/job/data/" || exit 1
fi
if ! $SSH -n 'test -f /workspace/job/model/model.safetensors'; then
  $SCP -r "$MODEL_DIR"/. "root@$HOST:/workspace/job/model/" || exit 1
fi
$SCP "$ROOT/embed_corpus.py" "root@$HOST:/workspace/job/" || exit 1

# 2. One process per GPU, each written as a job file and started fully detached (setsid),
#    so ssh returns at once. A shard that is running or done is left alone.
for SHARD in $(seq 0 $((SHARDS - 1))); do
  if $SSH -n "test -f /workspace/job/vectors/done_$SHARD || pgrep -f '[-]-shard $SHARD ' > /dev/null"; then
    log "shard $SHARD already running or done"
    continue
  fi
  $SSH "cat > /workspace/job/vectors/job_$SHARD.sh" <<JOB
cd /workspace/job
export PATH=\$HOME/.local/bin:\$PATH CUDA_VISIBLE_DEVICES=$SHARD
uv run --no-project --python 3.13 $(echo $PKGS) python embed_corpus.py --model model \\
  --corpus data/corpus.jsonl.gz --out vectors --shard $SHARD --shards $SHARDS \\
  > vectors/log_$SHARD.txt 2>&1
JOB
  $SSH -n "setsid nohup bash /workspace/job/vectors/job_$SHARD.sh > /dev/null 2>&1 < /dev/null &"
  log "started shard $SHARD"
done

# 3. Wait for every shard.
while true; do
  sleep 60
  DONE=$($SSH -n 'ls /workspace/job/vectors/done_* 2>/dev/null | wc -l')
  LAST=$($SSH -n 'for f in /workspace/job/vectors/log_*.txt; do grep -E "part|Error" "$f" | tail -1; done' | tr '\n' ' ')
  log "$DONE/$SHARDS shards done. $LAST"
  [ "$DONE" -ge "$SHARDS" ] && break
  $SSH -n 'grep -l -E "Traceback|Error" /workspace/job/vectors/log_*.txt' > /dev/null 2>&1 \
    && { log "a shard failed:"; $SSH -n 'tail -5 /workspace/job/vectors/log_*.txt'; exit 1; }
done

# 4. Copy back.
mkdir -p "$OUT"
$SCP "root@$HOST:/workspace/job/vectors/part_*.npz" "$OUT/" || exit 1
log "copied $(ls "$OUT"/part_*.npz | wc -l) parts to $OUT"
