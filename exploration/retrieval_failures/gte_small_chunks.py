"""Would smaller gte chunks lift the expected document into gte's top 50?

For each question the pool is gte's own top 50 (semantic search alone, on the stored
24,000-character chunks) plus the expected document and the pipeline's picks. Every pool
document is re-chunked at smaller sizes, embedded with gte-modernbert-base, and scored by
its best chunk. The expected document's rank in the pool is recorded for each size, next to
its rank under the stored chunks.

The pool is small, so this is optimistic: over the whole corpus, documents outside gte's
top 50 can also rise with smaller chunks.

Usage: uv run python exploration/retrieval_failures/gte_small_chunks.py \
    OUT.json FAILURES.json QUESTION_ID...
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.embedder import GteModernBertEmbedder
from train_rag.pipelines.retrieval import RetrievalPipeline
from train_rag.processing.chunking import RecursiveChunker
from train_rag.processing.normalize import normalize_documents, normalize_text
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.repository.question import load_questions

# Chunk size -> context borrowed from each neighbour, as the ingestion chunkers do.
SIZES = {4_000: 200, 2_000: 200, 1_000: 100}
OUT = Path(sys.argv[1])
FAILURES = Path(sys.argv[2])
QUESTION_IDS = sys.argv[3:]


def rank(target: float, others: list[float]) -> int:
    """Rank of the target score among itself and the others (1 = top)."""
    return 1 + sum(other > target for other in others)


async def gte_pools(questions: dict[str, str]) -> dict[str, list[dict[str, object]]]:
    """The top 50 of gte alone for every question, on the stored chunks."""
    pipeline = RetrievalPipeline.load(rerank=False)
    pools = {}
    try:
        for qid, text in questions.items():
            hits = await pipeline.semantic_search(normalize_text(text))
            pools[qid] = [
                {"doc_id": hit.doc_id, "gte_similarity": 1 - hit.distance} for hit in hits
            ]
            print(f"searched {len(pools)}/{len(questions)} {qid}", flush=True)
    finally:
        await pipeline.close()

    return pools


def main() -> None:
    """Re-chunk every pool, embed it and rank the expected document at each size."""
    records = {r["question_id"]: r for r in json.loads(FAILURES.read_text())}
    engine = get_engine()
    with Session(engine) as session:
        loaded = load_questions(session, EnterpriseRagBenchLoader.name, None, None)
        texts = {q.id: q.text for q in loaded if q.id in QUESTION_IDS}
    pools = asyncio.run(gte_pools(texts))

    members: dict[str, list[str]] = {}
    for qid in QUESTION_IDS:
        record = records[qid]
        ids = [record["expected"][0]["doc_id"]]
        ids += [hit["doc_id"] for hit in pools[qid]]
        ids += [pick["doc_id"] for pick in record["picked"]]
        members[qid] = list(dict.fromkeys(ids))
    unique = sorted({doc_id for ids in members.values() for doc_id in ids})
    with Session(engine) as session:
        internal = ids_of_doc_ids(session, EnterpriseRagBenchLoader.name, unique)
        documents = load_documents(session, sorted(internal))
        clean = {doc.doc_id: doc for doc in normalize_documents(documents)}
    print(f"{len(unique)} pool documents", flush=True)

    embedder = GteModernBertEmbedder(batch_size=32)
    queries = dict(
        zip(
            QUESTION_IDS,
            np.array(embedder.embed([normalize_text(texts[qid]) for qid in QUESTION_IDS])),
            strict=True,
        )
    )
    # Per size: every document's chunk vectors, embedded once and shared by all questions.
    vectors: dict[int, dict[str, np.ndarray]] = {}
    chunk_texts: dict[int, dict[str, list[str]]] = {}
    for size, context in SIZES.items():
        chunker = RecursiveChunker(size=size, context=context)
        chunk_texts[size] = {d: chunker.chunks(clean[d].title, clean[d].text) for d in clean}
        flat = [(d, t) for d, chunks in chunk_texts[size].items() for t in chunks]
        embedded = np.array(embedder.embed([t for _, t in flat]))
        vectors[size] = {}
        start = 0
        for doc_id in clean:
            count = len(chunk_texts[size][doc_id])
            vectors[size][doc_id] = embedded[start : start + count]
            start += count
        print(f"size {size:,}: {len(flat):,} chunks embedded", flush=True)

    results = []
    for qid in QUESTION_IDS:
        record = records[qid]
        target = record["expected"][0]
        stored = {hit["doc_id"]: hit["gte_similarity"] for hit in pools[qid]}
        others = [d for d in members[qid] if d != target["doc_id"] and d in stored]
        row: dict[str, object] = {
            "question_id": qid,
            "pool": len(others) + 1,
            "stored": {
                "target": target["gte_similarity"],
                "rank": rank(target["gte_similarity"], [stored[d] for d in others]),
                "top": max(stored.values()),
            },
        }
        for size in SIZES:
            best = {
                d: vectors[size][d] @ queries[qid] for d in members[qid] if d in clean
            }
            scores = {d: float(np.max(s)) for d, s in best.items()}
            position = int(np.argmax(best[target["doc_id"]]))
            row[str(size)] = {
                "target": scores[target["doc_id"]],
                "rank": rank(scores[target["doc_id"]], [scores[d] for d in others]),
                "top": max(scores[d] for d in others),
                "chunks": len(chunk_texts[size][target["doc_id"]]),
                "best_chunk": position,
                "best_text": chunk_texts[size][target["doc_id"]][position][:600],
            }
        results.append(row)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT}", flush=True)


main()
