"""Semantic benchmark questions: where the right document lands, base vs fine-tuned gte.

The benchmark run with the fine-tuned gte (run 31) gained only 2 net semantic questions over
the base pipeline (run 17), while the dev check promised far more. This replays both
pipelines on the 125 semantic questions, exactly as the evaluation searches (gte and BM25
top 50 each, fused with RRF to 50, reranked), and records each right document's rank in the
gte list, the fused list and the reranked list. Local, no Gemini.

Writes data/benchmark_semantic_check.json and prints the summary.

Usage:
    uv run python exploration/embedder_finetune/benchmark_semantic_check.py
"""

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import text

from train_rag.database import get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.fusion import reciprocal_rank_fusion
from train_rag.pipelines.retrieval import RERANK_DEPTH, RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress
from train_rag.reranker import Reranker

BASE_RUN = 17
FINE_TUNED_RUN = 31
OUTPUT = Path(__file__).parent / "data/benchmark_semantic_check.json"
# Rank buckets, upper bounds inclusive; "missing" means not in the list at all.
BUCKETS = ((5, "top 5"), (10, "6-10"), (20, "11-20"), (50, "21-50"))


def load_questions() -> list[dict[str, Any]]:
    """Read the semantic questions with both runs' verdicts.

    Returns:
        One record per question: id, text, right documents, correct in base and fine-tuned.
    """
    with get_engine().connect() as connection:
        rows = connection.execute(
            text(
                "select q.id, q.text, q.expected_doc_ids, b.correct as base_correct, "
                "f.correct as ft_correct from questions q "
                "join eval_results b on b.question_id = q.id and b.run_id = :base "
                "join eval_results f on f.question_id = q.id and f.run_id = :ft "
                "order by q.id"
            ),
            {"base": BASE_RUN, "ft": FINE_TUNED_RUN},
        ).all()

    return [
        {
            "id": row.id,
            "text": row.text,
            "expected": row.expected_doc_ids,
            "base_correct": row.base_correct,
            "ft_correct": row.ft_correct,
        }
        for row in rows
    ]


async def replay(
    pipeline: RetrievalPipeline, reranker: Reranker, question: str
) -> dict[str, list[str]]:
    """Run one pipeline's searches, fusion and reranking for a question.

    Args:
        pipeline: The loaded pipeline (its own reranker is not used).
        reranker: The cross-encoder shared by both pipelines.
        question: The question text.

    Returns:
        The doc ids of the gte top 50, the fused top 50 and the reranked 50, best first.
    """
    clean = normalize_text(question)
    semantic, keyword = await asyncio.gather(
        pipeline.semantic_search(clean), pipeline.keyword_search(clean)
    )
    fused = reciprocal_rank_fusion({"gte": semantic, "bm25": keyword}, RERANK_DEPTH)
    reranked = await pipeline.rerank(reranker, clean, await pipeline.describe(fused))

    return {
        "gte": [hit.doc_id for hit in semantic],
        "fused": [document.doc_id for document in fused],
        "reranked": [document.doc_id for document in reranked],
    }


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


def summarise(records: list[dict[str, Any]]) -> None:
    """Print where the right documents land, base vs fine-tuned, and how they moved.

    Args:
        records: Questions with both replayed rankings.
    """
    labels = [label for _, label in BUCKETS] + ["missing"]
    pairs = [(record, doc_id) for record in records for doc_id in record["expected"]]
    print(f"{len(records)} semantic questions, {len(pairs)} right documents\n")
    for stage in ("gte", "fused", "reranked"):
        print(f"{stage} list (base -> fine-tuned):")
        for label in labels:
            base = sum(bucket(doc_id, record["base"][stage]) == label for record, doc_id in pairs)
            tuned = sum(bucket(doc_id, record["ft"][stage]) == label for record, doc_id in pairs)
            print(f"  {label:8s} {base:4d} -> {tuned:4d}")
        print()

    moves: Counter[str] = Counter()
    for record, doc_id in pairs:
        before = bucket(doc_id, record["base"]["reranked"])
        after = bucket(doc_id, record["ft"]["reranked"])
        moves[f"{before} -> {after}"] += before != after
    print("reranked bucket changes for the same document:")
    for move, count in moves.most_common():
        if count:
            print(f"  {move:22s} {count}")

    never = [
        record["id"]
        for record in records
        if all(doc_id not in record["ft"]["fused"] for doc_id in record["expected"])
        and all(doc_id not in record["base"]["fused"] for doc_id in record["expected"])
    ]
    wrong_both = [r for r in records if not r["base_correct"] and not r["ft_correct"]]
    stuck_outside = [r for r in wrong_both if r["id"] in never]
    print(f"\nwrong in both runs: {len(wrong_both)}")
    print(f"  ... right document in neither fused 50: {len(stuck_outside)}")
    in_list = [r for r in wrong_both if r["id"] not in never]
    print(f"  ... right document in a fused 50 but still wrong: {len(in_list)}")


async def main() -> None:
    """Replay every semantic question with both pipelines, then print the summary."""
    records = load_questions()
    base = RetrievalPipeline.load(rerank=True)
    tuned = RetrievalPipeline.load(rerank=False, embedder=FineTunedGteEmbedder())
    reranker = base.reranker
    assert reranker is not None
    title = "semantic check: base vs fine-tuned gte"
    with RunProgress(get_engine(), title, len(records)) as progress:
        for record in records:
            record["base"] = await replay(base, reranker, record["text"])
            record["ft"] = await replay(tuned, reranker, record["text"])
            progress.advance("replaying", 1)
    await base.close()
    await tuned.close()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(records, indent=1))
    summarise(records)


if __name__ == "__main__":
    asyncio.run(main())
