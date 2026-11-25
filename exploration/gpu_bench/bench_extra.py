"""Follow-up measurements: the query embedder on CPU, and lighter reranking settings on GPU.

Run from the main checkout, like bench.py:
    uv run python <workspace>/exploration/gpu_bench/bench_extra.py --out <workspace>/exploration/gpu_bench/results_extra.json
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


def main() -> None:
    """Embed questions on CPU with 1, 2 and 4 threads; rerank 50/100 candidates at 512/1,024 tokens."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    questions = pd.read_parquet(QUESTIONS)["question"].tolist()
    out: dict[str, object] = {}

    # query embedding on CPU, one question at a time (what a request does)
    emb = SentenceTransformer(EMBEDDER, device="cpu")
    emb.encode(questions[:4])
    cpu = {}
    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        sample = questions[:60]
        t = time.perf_counter()
        for q in sample:
            emb.encode([q], normalize_embeddings=True)
        s = time.perf_counter() - t
        cpu[f"threads_{threads}"] = {"ms_per_question": 1000 * s / len(sample), "questions_per_s": len(sample) / s}
        print("cpu embed", threads, cpu[f"threads_{threads}"], flush=True)
    out["query_embedding_cpu"] = cpu
    del emb

    # reranking settings on GPU
    rng = random.Random(7)
    docs = pd.read_parquet(DOCUMENTS, columns=["title", "content"]).sample(4000, random_state=7)
    texts = [f"{t}\n{c}" for t, c in zip(docs["title"], docs["content"])]
    batches = [(questions[i], rng.sample(texts, 100)) for i in range(12)]
    rr = {}
    for tokens in (512, 1024):
        model = CrossEncoder(RERANKER, max_length=tokens, model_kwargs={"torch_dtype": torch.float16}, device="cuda")
        model.predict([(batches[0][0], d) for d in batches[0][1][:16]], batch_size=8)
        for n in (50, 100):
            times = []
            for q, cands in batches:
                pairs = [(q, d) for d in sorted(cands[:n], key=len)]
                torch.cuda.synchronize()
                t = time.perf_counter()
                model.predict(pairs, batch_size=8)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            key = f"{n}_docs_{tokens}_tokens"
            rr[key] = {"ms_per_question_median": 1000 * statistics.median(times),
                       "questions_per_s": len(times) / sum(times), "pairs_per_s": n * len(times) / sum(times)}
            print("rerank", key, rr[key], flush=True)
        del model
        torch.cuda.empty_cache()
    out["rerank_settings_gpu"] = rr
    args.out.write_text(json.dumps(out, indent=1))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
