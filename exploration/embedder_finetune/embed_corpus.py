r"""Embed the whole corpus with a fine-tuned model, on a rented GPU (no database needed).

Reads corpus.jsonl.gz (from export_corpus.py). Several GPUs work in parallel: each process
takes every n-th chunk (--shard i --shards n, one process per GPU). Chunks are embedded in
parts of PART chunks; each finished part is saved as part_<shard>_<n>.npz (float16 unit
vectors plus the table and row id of each), so a restart skips finished parts.

Same settings as production: half precision, up to 8,192 tokens, unit vectors.

Usage (one process per GPU):
    CUDA_VISIBLE_DEVICES=0 python embed_corpus.py --model runs/X/final \\
        --corpus data/corpus.jsonl.gz --out vectors --shard 0 --shards 4
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PART = 20_000
MAX_TOKENS = 8192


def read_shard(corpus: Path, shard: int, shards: int) -> list[dict[str, object]]:
    """The chunks of one shard: every `shards`-th line, starting at `shard`.

    Args:
        corpus: The exported corpus.
        shard: This process's shard.
        shards: Number of processes.

    Returns:
        The chunks, each with its table, id and text.
    """
    with gzip.open(corpus, "rt") as lines:
        return [json.loads(line) for index, line in enumerate(lines) if index % shards == shard]


def main() -> None:
    """Embed this shard part by part, skipping parts already saved."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    # 1. This shard's chunks and the model, in half precision like production.
    args.out.mkdir(parents=True, exist_ok=True)
    chunks = read_shard(args.corpus, args.shard, args.shards)
    model = SentenceTransformer(str(args.model), model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = MAX_TOKENS
    parts = range(0, len(chunks), PART)
    print(f"shard {args.shard}: {len(chunks):,} chunks in {len(parts)} parts", flush=True)

    # 2. Embed part by part; a saved part is never redone.
    start = time.time()

    for number, first in enumerate(parts):
        path = args.out / f"part_{args.shard}_{number}.npz"
        if path.exists():
            continue

        part = chunks[first : first + PART]
        vectors = model.encode(
            [str(chunk["text"]) for chunk in part],
            batch_size=args.batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        np.savez(
            path,
            vectors=vectors.astype(np.float16),
            tables=np.array([chunk["table"] for chunk in part]),
            ids=np.array([chunk["id"] for chunk in part], dtype=np.int64),
        )

        minutes = (time.time() - start) / 60
        print(f"shard {args.shard}: part {number + 1}/{len(parts)} ({minutes:.1f} min)", flush=True)

    # 3. A marker, so the caller knows this shard is complete.
    (args.out / f"done_{args.shard}").write_text(str(len(chunks)))


if __name__ == "__main__":
    main()
