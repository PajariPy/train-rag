"""Rerank the full union of the gte and BM25 top 50 instead of the fused top 50.

The semantic check (benchmark_semantic_check.py) found that fusion keeps only 50 of the up to
100 documents the two searches return, and cut 8 right documents the fine-tuned gte had
found. The reranker puts about 90% of the right documents it sees into the top 5. This
replays the 125 semantic questions with both embedders and reranks two candidate lists, the
fused 50 (today) and the whole union, on the same text the pipeline uses (nothing cut).
Local, no Gemini.

Writes data/union_rerank_check.json and prints the summary.

Usage:
    uv run python exploration/embedder_finetune/union_rerank_check.py
"""

import asyncio
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

from benchmark_semantic_check import load_questions

from train_rag.database import get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.fusion import reciprocal_rank_fusion
from train_rag.pipelines.retrieval import CANDIDATES, RERANK_DEPTH, RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress
from train_rag.reranker import Reranker

OUTPUT = Path(__file__).parent / "data/union_rerank_check.json"
# Rank buckets, upper bounds inclusive; the union can hold up to 100 documents.
BUCKETS = ((5, "top 5"), (10, "6-10"), (20, "11-20"), (50, "21-50"), (100, "51-100"))
LABELS = tuple(label for _, label in BUCKETS) + ("missing",)
FINAL = 5


def bucket(doc_id: str, ranking: list[str]) -> str:
    """Name the rank bucket of a document in a ranking.

    Args:
        doc_id: The document.
        ranking: Doc ids, best first.

    Returns:
        A bucket label from `BUCKETS`, or "missing".
    """
    if doc_id not in ranking:
        return "missing"
    rank = ranking.index(doc_id) + 1

    return next(label for bound, label in BUCKETS if rank <= bound)


async def replay(
    pipeline: RetrievalPipeline, reranker: Reranker, question: str
) -> dict[str, Any]:
    """Search once, then rerank the fused 50 and the whole union for one question.

    Args:
        pipeline: The loaded pipeline (its own reranker is not used).
        reranker: The cross-encoder shared by both pipelines.
        question: The question text.

    Returns:
        The reranked doc ids of the fused 50 and of the union, best first, the union's
        size and the seconds each reranking took.
    """
    clean = normalize_text(question)
    semantic, keyword = await asyncio.gather(
        pipeline.semantic_search(clean), pipeline.keyword_search(clean)
    )
    rankings = {"gte": semantic, "bm25": keyword}
    union = await pipeline.describe(reciprocal_rank_fusion(rankings, 2 * CANDIDATES))
    fused_ids = {
        document.doc_id for document in reciprocal_rank_fusion(rankings, RERANK_DEPTH)
    }
    fused = [document for document in union if document.doc_id in fused_ids]

    started = time.perf_counter()
    fused_reranked = await pipeline.rerank(reranker, clean, fused)
    fused_seconds = time.perf_counter() - started
    started = time.perf_counter()
    union_reranked = await pipeline.rerank(reranker, clean, union)
    union_seconds = time.perf_counter() - started

    return {
        "fused": [document.doc_id for document in fused_reranked],
        "union": [document.doc_id for document in union_reranked],
        "union_size": len(union),
        "fused_seconds": fused_seconds,
        "union_seconds": union_seconds,
    }


def summarise(records: list[dict[str, Any]]) -> None:
    """Print where the right documents land with each embedder and candidate list.

    Args:
        records: Questions with both embedders' replays.
    """
    pairs = [(record, doc_id) for record in records for doc_id in record["expected"]]
    print(f"{len(records)} semantic questions, {len(pairs)} right documents\n")
    columns = [(embedder, stage) for embedder in ("base", "ft") for stage in ("fused", "union")]
    print(f"{'':10s}" + "".join(f"{e + ' ' + s:>14s}" for e, s in columns))
    for label in LABELS:
        counts = [
            sum(bucket(doc_id, record[e][s]) == label for record, doc_id in pairs)
            for e, s in columns
        ]
        print(f"{label:10s}" + "".join(f"{count:14d}" for count in counts))

    print("\nquestions with a right document in the final 5:")
    for embedder, stage in columns:
        hits = sum(
            any(doc_id in record[embedder][stage][:FINAL] for doc_id in record["expected"])
            for record in records
        )
        print(f"  {embedder} {stage:6s} {hits:4d} / {len(records)}")

    for embedder in ("base", "ft"):
        gained = lost = 0
        for record in records:
            before = any(d in record[embedder]["fused"][:FINAL] for d in record["expected"])
            after = any(d in record[embedder]["union"][:FINAL] for d in record["expected"])
            gained += after and not before
            lost += before and not after
        sizes = [record[embedder]["union_size"] for record in records]
        fused_s = median(record[embedder]["fused_seconds"] for record in records)
        union_s = median(record[embedder]["union_seconds"] for record in records)
        print(
            f"\n{embedder}: union vs fused, final 5 gained {gained}, lost {lost}; "
            f"union size median {median(sizes):.0f}; "
            f"rerank median {fused_s:.1f} s (fused) vs {union_s:.1f} s (union)"
        )


async def main() -> None:
    """Replay the semantic questions not yet in the output file, then print the summary.

    The output is written after every question, so a stopped run picks up where it left off.
    """
    records = load_questions()
    done: dict[str, Any] = (
        {record["id"]: record for record in json.loads(OUTPUT.read_text())}
        if OUTPUT.exists()
        else {}
    )
    todo = [record for record in records if record["id"] not in done]
    print(f"{len(done)} questions already replayed, {len(todo)} to go")
    if todo:
        base = RetrievalPipeline.load(rerank=True)
        tuned = RetrievalPipeline.load(rerank=False, embedder=FineTunedGteEmbedder())
        reranker = base.reranker
        assert reranker is not None
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        title = "union rerank check: fused 50 vs union, base and fine-tuned gte"
        with RunProgress(get_engine(), title, len(todo)) as progress:
            for record in todo:
                record["base"] = await replay(base, reranker, record["text"])
                record["ft"] = await replay(tuned, reranker, record["text"])
                done[record["id"]] = record
                OUTPUT.write_text(json.dumps(list(done.values()), indent=1))
                progress.advance("replaying", 1)
        await base.close()
        await tuned.close()
    summarise([done[record["id"]] for record in records])


if __name__ == "__main__":
    asyncio.run(main())
