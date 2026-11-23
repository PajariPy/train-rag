#!/usr/bin/env bash
# Mine one part of the corpus on a rented GPU, with no laptop needed (runs on the machine).
#
# Started detached by vast_mine.sh in /workspace/mine, which holds: job.env (PART, PRICE,
# GRACE_HOURS, MAX_HOURS, PEERS), part.json (the shards' sizes), shards/*.jsonl.gz, adapter/,
# merge_lora.py, generate_vllm.py and mine_status.py.
#
# 1. Starts the status page (mine_status.py, port 8000).
# 2. Merges the LoRA once, then runs generate_vllm.py on every shard in turn; each shard writes
#    out/<shard>.jsonl (resumable) and, when complete, out/<shard>.done.
# 3. Then waits up to GRACE_HOURS for the laptop's fetch (which writes `fetched` and destroys
#    the machine). With no fetch, or on any failure, or after MAX_HOURS, the machine stops
#    itself: GPU billing ends, the disk stays, and fetch can start it again later.
#
# The phase is in the file `phase` (setup, merge, mining shard_NN, finished, failed, stopping),
# an error in `error`. Never edit this script on the machine while it runs.
set -u
cd /workspace/mine || exit 1
# shellcheck disable=SC1091
source job.env
export PATH=$HOME/.local/bin:$PATH
mkdir -p out logs
date +%s > started
VLLM=0.33.0
MERGE_PKGS=(--with "transformers @ git+https://github.com/huggingface/transformers" --with peft
  --with accelerate --with torch==2.15.0 --with torchvision --with pillow)

phase() { echo "$1" > phase; }

stop_self() {  # Vast's per-instance key may stop this instance; the disk is kept
  phase stopping
  echo "$(date -u +%H:%M:%S) stopping: $1" >> stop.log
  for _ in 1 2 3; do
    uvx vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> stop.log 2>&1 && exit 0
    sleep 30
  done
  echo "could not stop itself" >> stop.log
  exit 1
}

setsid nohup python3 mine_status.py > status.log 2>&1 < /dev/null &

# The safety deadline: stop after MAX_HOURS whatever happens, unless fetched.
(
  sleep $((MAX_HOURS * 3600))
  [ -f fetched ] || stop_self "MAX_HOURS ($MAX_HOURS h) reached"
) &

phase setup
if ! command -v uv > /dev/null; then
  (curl -LsSf https://astral.sh/uv/install.sh | sh) > setup.log 2>&1
fi
# Check now, not after 10 h, that the machine can stop itself.
if [ -z "${CONTAINER_ID:-}" ] || [ -z "${CONTAINER_API_KEY:-}" ]; then
  echo "no CONTAINER_ID / CONTAINER_API_KEY: the machine can't stop itself" > error
fi

phase merge
if [ ! -f merged/config.json ]; then
  if ! uv run --no-project --python 3.13 "${MERGE_PKGS[@]}" python merge_lora.py adapter merged \
      >> setup.log 2>&1; then
    echo "merge failed (setup.log)" > error
    phase failed
    stop_self "merge failed"
  fi
fi

for gz in shards/shard_*.jsonl.gz; do
  name=$(basename "$gz" .jsonl.gz)
  [ -f "out/$name.done" ] && continue
  phase "mining $name"
  [ -f "prompts_$name.jsonl" ] || gunzip -c "$gz" > "prompts_$name.jsonl"
  uv run --no-project --python 3.12 --with vllm==$VLLM python generate_vllm.py merged \
    "prompts_$name.jsonl" "out/$name.jsonl" >> "logs/$name.log" 2>&1
  code=$?
  if [ $code -ne 0 ]; then
    echo "$name: generate_vllm.py exited with $code (logs/$name.log)" > error
    phase failed
    stop_self "$name failed"
  fi
  touch "out/$name.done"
  rm -f "prompts_$name.jsonl"
done

phase finished
date +%s > finished
deadline=$(( $(date +%s) + GRACE_HOURS * 3600 ))
while [ ! -f fetched ] && [ "$(date +%s)" -lt "$deadline" ]; do sleep 60; done
[ -f fetched ] || stop_self "no fetch within $GRACE_HOURS h of finishing"
