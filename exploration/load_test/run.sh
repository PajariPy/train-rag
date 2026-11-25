#!/usr/bin/env bash
# Load test of the stand-in query service on 1 CPU and 2 GB (like a small Fargate task).
# Runs twice: BM25 index fully in memory, and memory-mapped. Run from the project root:
#   bash exploration/load_test/run.sh
set -euo pipefail
HERE=exploration/load_test
INDEX="$PWD/data/bm25/all"
docker build -q -t train-rag-loadtest "$HERE" >/dev/null

for mode in inmemory mmap; do
  mmap=0; [ "$mode" = mmap ] && mmap=1
  docker rm -f loadtest >/dev/null 2>&1 || true
  docker run -d --name loadtest --cpus=1 --memory=2g --memory-swap=2g -p 127.0.0.1:8800:8000 \
    -v "$INDEX":/index:ro -e BM25_MMAP=$mmap train-rag-loadtest >/dev/null
  echo "== $mode: loading the index"
  for _ in $(seq 1 180); do
    curl -sf http://127.0.0.1:8800/health >/dev/null && break
    if [ "$(docker inspect -f '{{.State.Running}}' loadtest)" != true ]; then
      echo "server died while loading (OOMKilled=$(docker inspect -f '{{.State.OOMKilled}}' loadtest))"
      docker logs loadtest 2>&1 | tail -5
      continue 2
    fi
    sleep 1
  done
  uv run python "$HERE/loadgen.py" --container loadtest --out "$HERE/results_$mode.json" "$@"
  echo "OOMKilled=$(docker inspect -f '{{.State.OOMKilled}}' loadtest)"
done
docker rm -f loadtest >/dev/null
