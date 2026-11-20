"""Merge HyDE and rewrites into the search query instead of adding searches.

expansion_check.py showed that extra searches with rewritten queries lift semantic dev
questions (top 5 61.0% → 69.5%) but make the reranker read ~390 documents per question.
Here the expansions are merged *before* the search, so there is still one gte search and
one BM25 search and the reranker still reads about 100 documents:

- gte: the mean of the question's and the expansions' vectors (normalized again), or one
  vector of the question with the rewrites appended;
- BM25: the question with the rewrites appended (the keyword rewrite alone is already in
  expansion_check.json).

Reuses the Gemini expansions and reranker scores from expansion_check.py; only documents not
scored there are reranked. Local and free.

Writes data/expansion_merge.json (resumable), then prints the summary.

Usage:
    uv run python exploration/query_expansion/expansion_merge.py
"""

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import numpy as np
from expansion_check import DATA, DEV, EXPANDED_TYPE, EXPANSIONS, OUTPUT

from train_rag.database import get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.fusion import RRF_K, best_per_document, reciprocal_rank_fusion
from train_rag.models.retrieval import SearchHit
from train_rag.pipelines.retrieval import CANDIDATES, RERANK_DEPTH, RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress

MERGED = DATA / "expansion_merge.json"
# gte variants: name -> (weight of the question, which expansions are averaged with it).
MEAN_VARIANTS = {
    "mean_rw": ("equal", ["rw1", "rw2", "rw3", "rw4"]),
    "mean_hyde": ("equal", ["hyde"]),
    "mean_all": ("equal", ["hyde", "rw1", "rw2", "rw3", "rw4"]),
    "mean_rw_q50": ("half", ["rw1", "rw2", "rw3", "rw4"]),
}


def mean_vector(
    vectors: dict[str, list[float]], question_weight: str, names: list[str]
) -> list[float]:
    """Average the question's vector with some expansions' vectors, then normalize.

    Args:
        vectors: Text name -> normalized vector ("q", "hyde", "rw1", ...).
        question_weight: "equal" (every vector counts the same) or "half" (the question
            weighs as much as all expansions together).
        names: The expansions to average with the question.

    Returns:
        The normalized mean vector.
    """
    question = np.array(vectors["q"])
    others = np.mean([vectors[name] for name in names], axis=0)
    if question_weight == "half":
        mean = 0.5 * question + 0.5 * others
    else:
        mean = (question + len(names) * others) / (1 + len(names))

    return (mean / np.linalg.norm(mean)).tolist()


async def vector_search(pipeline: RetrievalPipeline, vector: Sequence[float]) -> list[SearchHit]:
    """Search every source's chunks with a given vector, like the pipeline's gte search.

    Args:
        pipeline: The loaded pipeline.
        vector: The query vector.

    Returns:
        The best chunk of up to `CANDIDATES` documents, closest first.
    """
    per_source = await asyncio.gather(
        *(pipeline.search_source(source, list(vector)) for source in pipeline.sources)
    )

    return best_per_document([hit for hits in per_source for hit in hits], CANDIDATES)


async def replay(
    pipeline: RetrievalPipeline, question: str, expansion: dict[str, Any], scored: dict[str, float]
) -> dict[str, Any]:
    """Run the merged searches for one question and score their new candidates.

    Args:
        pipeline: The loaded pipeline with its reranker.
        question: The question text.
        expansion: Its HyDE passage and rewrites.
        scored: Reranker scores already known from expansion_check.py.

    Returns:
        The doc ids of each merged search and the scores of the documents new to it.
    """
    texts = {"q": question, "hyde": expansion["hyde"]}
    texts |= {f"rw{i}": query for i, query in enumerate(expansion["rewrites"], 1)}
    concat = " ".join([question, *expansion["rewrites"]])
    names = [*texts, "concat"]
    embedded = pipeline.embedder.embed(
        [normalize_text(texts[n]) if n != "concat" else normalize_text(concat) for n in names]
    )
    vectors = dict(zip(names, embedded, strict=True))

    lists: dict[str, list[SearchHit]] = {
        f"gte:{name}": await vector_search(pipeline, mean_vector(vectors, weight, parts))
        for name, (weight, parts) in MEAN_VARIANTS.items()
    }
    lists["gte:concat"] = await vector_search(pipeline, vectors["concat"])
    lists["bm25:concat"] = await pipeline.keyword_search(normalize_text(concat))

    every = reciprocal_rank_fusion(lists, CANDIDATES * len(lists))
    new = [document for document in every if document.doc_id not in scored]
    scores: dict[str, float] = {}
    if new:
        assert pipeline.reranker is not None
        described = await pipeline.describe(new)
        reranked = await pipeline.rerank(pipeline.reranker, normalize_text(question), described)
        scores = {document.doc_id: document.rerank_score or 0.0 for document in reranked}

    return {
        "lists": {name: [hit.doc_id for hit in hits] for name, hits in lists.items()},
        "scores": scores,
    }


def summarise(rows: list[dict[str, Any]], base: dict[str, Any], merged: dict[str, Any]) -> None:
    """Print, per gte and BM25 query, how the merged searches do on the semantic dev questions.

    Args:
        rows: Semantic dev questions.
        base: Question id -> expansion_check.py record (lists and scores).
        merged: Question id -> merged lists and the scores of their new documents.
    """
    gte_options = ["gte:q", *[f"gte:{name}" for name in MEAN_VARIANTS], "gte:concat"]
    bm25_options = ["bm25:q", "bm25:rw1", "bm25:concat"]
    n = len(rows)
    print(f"semantic dev ({n} questions), one gte + one BM25 search")
    print(
        f"  {'gte query':14s} {'BM25 query':12s} {'in union':>9s} {'union top 5':>12s}"
        f" {'docs':>5s} {'fused 50 top 5':>15s}"
    )
    for gte in gte_options:
        for bm25 in bm25_options:
            found = top5 = fused5 = size = 0
            for row in rows:
                lists = base[row["id"]]["lists"] | merged[row["id"]]["lists"]
                scores = base[row["id"]]["scores"] | merged[row["id"]]["scores"]
                pool = set(lists[gte]) | set(lists[bm25])
                order = sorted(pool, key=lambda doc_id: scores[doc_id], reverse=True)
                fused = fuse_ids([lists[gte], lists[bm25]])[:RERANK_DEPTH]
                fused_order = sorted(fused, key=lambda doc_id: scores[doc_id], reverse=True)
                positive = row["positive"]
                size += len(pool)
                found += positive in pool
                top5 += positive in order[:5]
                fused5 += positive in fused_order[:5]
            print(
                f"  {gte[4:]:14s} {bm25[5:]:12s} {found / n:9.1%} {top5 / n:12.1%} "
                f"{size / n:5.0f} {fused5 / n:15.1%}"
            )


def fuse_ids(rankings: list[list[str]]) -> list[str]:
    """Reciprocal rank fusion of doc id lists, as the pipeline fuses its searches.

    Args:
        rankings: Doc ids, best first, one list per search.

    Returns:
        The doc ids by fused score, best first.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (RRF_K + rank)

    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


async def main() -> None:
    """Run the merged searches for every semantic dev question, then print the summary."""
    rows = [json.loads(line) for line in DEV.read_text().splitlines()]
    rows = [row for row in rows if row["question_type"] == EXPANDED_TYPE]
    expansions = json.loads(EXPANSIONS.read_text())
    base = json.loads(OUTPUT.read_text())
    merged: dict[str, Any] = json.loads(MERGED.read_text()) if MERGED.exists() else {}
    todo = [row for row in rows if row["id"] not in merged]
    print(f"{len(merged)} questions already run, {len(todo)} to go")
    if todo:
        pipeline = RetrievalPipeline.load(rerank=True, embedder=FineTunedGteEmbedder())
        title = "query expansion: merged queries (dev semantic, local)"
        with RunProgress(get_engine(), title, len(todo), unit="questions") as progress:
            for row in todo:
                merged[row["id"]] = await replay(
                    pipeline, row["question"], expansions[row["id"]], base[row["id"]]["scores"]
                )
                MERGED.write_text(json.dumps(merged))
                progress.advance("searching", 1, note="local, $0")
        await pipeline.close()
    summarise(rows, base, merged)


if __name__ == "__main__":
    asyncio.run(main())
