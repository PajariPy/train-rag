"""Gemini embeddings against gte and our reranker on the 46 semantic failures.

For each failure of run 15, its candidates (`pools_run15.json`: gte's top 50, the right
document and the documents the run showed, about 53) are ranked three ways:

- gte: its own search order (the right document is outside its top 50 when missed);
- reranker: gte-reranker-modernbert-base on full text (`rerank_pilot_scores.json`);
- Gemini: gemini-embedding-2, query vs best chunk, the same texts gte embedded.

The question is which teacher would pick hard negatives and spot the right document
better. Vectors share the cache of compare.py. About $0.75.

Usage:
    uv run --env-file .env python exploration/gemini_embedding/failures.py
"""

import json
from pathlib import Path

import numpy as np
from compare import embed_all, load_texts, unit
from scipy.stats import spearmanr

from train_rag.processing.normalize import normalize_text

ROOT = Path(__file__).parent.parent
POOLS = ROOT / "retrieval_failures/data/pools_run15.json"
RERANK = ROOT / "fact_extraction/data/rerank_pilot_scores.json"
RESULT = Path(__file__).parent / "data/failures.json"


def rank_of(scores: dict[str, float], doc_id: str) -> int:
    """Rank of one document by score among the candidates.

    Args:
        scores: Score per candidate.
        doc_id: The document.

    Returns:
        1 + the number of candidates scoring higher.
    """
    return 1 + sum(value > scores[doc_id] for value in scores.values())


def main() -> None:
    """Embed the failures' candidates with Gemini, rank the right document three ways, print."""
    pools = json.loads(POOLS.read_text())["pools"]
    rerank = json.loads(RERANK.read_text())
    doc_ids = sorted({doc_id for pool in pools.values() for doc_id in pool["members"]})
    chunks = load_texts(doc_ids)
    items = {
        f"q:{question_id}": (normalize_text(pool["question"]), "RETRIEVAL_QUERY")
        for question_id, pool in pools.items()
    }
    items |= {
        f"d:{doc_id}:{position}": (chunk_text, "RETRIEVAL_DOCUMENT")
        for doc_id, position, chunk_text in chunks
    }
    vectors = embed_all(items)

    chunk_keys: dict[str, list[str]] = {}
    for doc_id, position, _ in chunks:
        chunk_keys.setdefault(doc_id, []).append(f"d:{doc_id}:{position}")

    rows = []
    for question_id, pool in sorted(pools.items()):
        members = [doc_id for doc_id in pool["members"] if doc_id in chunk_keys]
        query = unit([vectors[f"q:{question_id}"]])[0]
        gemini = {
            doc_id: float(max(unit([vectors[key] for key in chunk_keys[doc_id]]) @ query))
            for doc_id in members
        }
        reranker = {doc_id: rerank[question_id]["rerank_text"][doc_id] for doc_id in members}
        gte_order = pool["gte_top50"]
        expected = pool["expected"]
        rows.append(
            {
                "question": question_id,
                "candidates": len(members),
                "gte rank": gte_order.index(expected) + 1 if expected in gte_order else None,
                "reranker rank": rank_of(reranker, expected),
                "gemini rank": rank_of(gemini, expected),
                "gemini vs reranker (Spearman)": float(
                    spearmanr(
                        [gemini[doc] for doc in members], [reranker[doc] for doc in members]
                    ).statistic
                ),
                "gemini width": max(gemini.values()) - min(gemini.values()),
                "gemini gap to rank 1": max(gemini.values()) - gemini[expected],
            }
        )
    RESULT.write_text(json.dumps(rows, indent=1))

    print(
        f"\n{len(rows)} failures, median "
        f"{np.median([row['candidates'] for row in rows]):.0f} candidates"
    )
    print(f"{'ranked by':10s} {'top 1':>6s} {'top 5':>6s} {'top 10':>6s} {'median rank':>12s}")
    for name in ("gte", "reranker", "gemini"):
        ranks = [row[f"{name} rank"] or 99 for row in rows]
        print(
            f"{name:10s} {sum(rank <= 1 for rank in ranks):6d} "
            f"{sum(rank <= 5 for rank in ranks):6d} "
            f"{sum(rank <= 10 for rank in ranks):6d} {np.median(ranks):12.0f}"
        )
    both = sum(row["reranker rank"] <= 5 and row["gemini rank"] <= 5 for row in rows)
    either = sum(row["reranker rank"] <= 5 or row["gemini rank"] <= 5 for row in rows)
    print(f"top 5 by both teachers: {both}, by either: {either}")
    print(
        "Gemini vs reranker order, median Spearman: "
        f"{np.median([row['gemini vs reranker (Spearman)'] for row in rows]):.2f}"
    )


if __name__ == "__main__":
    main()
