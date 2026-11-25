"""Measure how fast the embedder and the reranker run on the local GPU.

Uses the real models with the pipeline's settings (float16, sentence-transformers), real
benchmark questions and real documents:

    query    embed questions (short) at several batch sizes            -> questions/s
    rerank   score 100 documents per question, cut at 1,024 tokens       -> pairs/s, questions/s
    ingest   embed whole documents (up to 8,192 tokens), shortest first  -> documents/s

Run from the main checkout (it has the models and the environment):
    cd ~/projects/rag-implementation/train-rag
    uv run python <workspace>/exploration/gpu_bench/bench.py --out <workspace>/exploration/gpu_bench/results.json
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBEDDER = "models/gte-modernbert-ft"
RERANKER = "Alibaba-NLP/gte-reranker-modernbert-base"
QUESTIONS = "data/enterprise_rag_bench/data/questions/test.parquet"
DOCUMENTS = "data/enterprise_rag_bench/data/documents/test.parquet"
RERANK_TOKENS = 1024
CANDIDATES = 100


def timed(fn) -> float:
    """Run fn on the GPU and return the seconds it took."""
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t


def main() -> None:
    """Run the three measurements and save them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", type=int, default=600, help="documents for the ingestion measurement")
    ap.add_argument("--rerank-questions", type=int, default=15)
    args = ap.parse_args()

    rng = random.Random(7)
    questions = pd.read_parquet(QUESTIONS)["question"].tolist()
    docs = pd.read_parquet(DOCUMENTS, columns=["title", "content"]).sample(4000, random_state=7)
    texts = [f"{t}\n{c}" for t, c in zip(docs["title"], docs["content"])]
    out: dict[str, object] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}

    # ---- 1. query embedding ----
    emb = SentenceTransformer(EMBEDDER, model_kwargs={"torch_dtype": torch.float16}, device="cuda")
    emb.max_seq_length = 8192
    tok = emb.tokenizer
    q_tokens = [len(tok(q)["input_ids"]) for q in questions[:500]]
    emb.encode(questions[:64], batch_size=32)  # warm-up
    query = {"question_tokens_median": statistics.median(q_tokens)}
    for bs in (1, 8, 32, 64):
        sample = questions[: max(128, bs * 8)]
        s = timed(lambda: emb.encode(sample, batch_size=bs, normalize_embeddings=True))
        query[f"batch_{bs}"] = {"questions_per_s": len(sample) / s, "ms_per_batch": 1000 * s / (len(sample) / bs)}
    out["query_embedding"] = query
    print("query embedding:", json.dumps(query, indent=1), flush=True)

    # ---- 3. document embedding (ingestion) ----
    sample = texts[: args.docs]
    d_tokens = sorted(min(8192, len(tok(t, truncation=False)["input_ids"])) for t in sample)
    order = sorted(sample, key=len)
    emb.encode(order[:8], batch_size=4)  # warm-up
    s = timed(lambda: emb.encode(order, batch_size=4, normalize_embeddings=True))
    out["document_embedding"] = {
        "documents": len(sample), "seconds": s, "documents_per_s": len(sample) / s,
        "tokens_per_s": sum(d_tokens) / s,
        "doc_tokens_median": statistics.median(d_tokens),
        "doc_tokens_p90": d_tokens[int(0.9 * len(d_tokens))],
        "share_cut_at_8192": sum(t >= 8192 for t in d_tokens) / len(d_tokens),
    }
    print("document embedding:", json.dumps(out["document_embedding"], indent=1), flush=True)
    del emb
    torch.cuda.empty_cache()

    # ---- 2. reranking ----
    rr = CrossEncoder(RERANKER, max_length=RERANK_TOKENS, model_kwargs={"torch_dtype": torch.float16}, device="cuda")
    pair_tokens = [min(RERANK_TOKENS, len(rr.tokenizer(questions[0], t)["input_ids"])) for t in texts[:500]]
    rerank = {"pair_tokens_median": statistics.median(pair_tokens),
              "share_cut_at_1024": sum(t >= RERANK_TOKENS for t in pair_tokens) / len(pair_tokens)}
    batches = [(questions[i], rng.sample(texts, CANDIDATES)) for i in range(args.rerank_questions)]
    rr.predict([(batches[0][0], d) for d in batches[0][1][:16]], batch_size=8)  # warm-up
    for bs in (8, 16, 32):
        times = []
        for q, cands in batches:
            pairs = [(q, d) for d in sorted(cands, key=len)]  # shortest first, like the pipeline
            times.append(timed(lambda: rr.predict(pairs, batch_size=bs)))
        total = sum(times)
        rerank[f"batch_{bs}"] = {
            "pairs_per_s": CANDIDATES * len(batches) / total,
            "questions_per_s": len(batches) / total,
            "ms_per_question_median": 1000 * statistics.median(times),
        }
        print(f"rerank batch {bs}:", rerank[f"batch_{bs}"], flush=True)
    out["rerank"] = rerank

    args.out.write_text(json.dumps(out, indent=1))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
