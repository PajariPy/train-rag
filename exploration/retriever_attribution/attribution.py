"""Which retriever brings up the right document: BM25, gte, or both?

The full evaluation (hybrid-rerank: run 7 for basic, 17 for semantic, 19-26 for the other
types) kept only the final 5 documents, not the two rankings. Search is deterministic, so
this replays both rankings for every question, exactly as the pipeline does (top 50 each,
fused with RRF to 50 for the reranker), and records where each right document was.

Writes data/attribution.json (resumable) and prints the summary.

Usage:
    uv run python exploration/retriever_attribution/attribution.py
"""

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import text

from train_rag.database import get_engine
from train_rag.fusion import reciprocal_rank_fusion
from train_rag.pipelines.retrieval import RERANK_DEPTH, RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress

FULL_EVAL_RUNS = (7, 17, 19, 20, 21, 22, 23, 24, 25, 26)
OUTPUT = Path(__file__).parent / "data/attribution.json"
FINAL = 5


def load_questions() -> list[dict[str, Any]]:
    """Read every question of the full evaluation with its final documents and verdict.

    Returns:
        One record per question: id, type, text, right documents, final 5, correct.
    """
    with get_engine().connect() as connection:
        rows = connection.execute(
            text(
                "select q.id, q.question_type, q.text, q.expected_doc_ids, "
                "r.retrieved_doc_ids, r.correct, r.run_id from eval_results r "
                "join questions q on q.id = r.question_id where r.run_id = any(:runs)"
            ),
            {"runs": list(FULL_EVAL_RUNS)},
        ).all()

    return [
        {
            "id": row.id,
            "type": row.question_type,
            "text": row.text,
            "expected": row.expected_doc_ids,
            "final": row.retrieved_doc_ids[:FINAL],
            "correct": row.correct,
            "run": row.run_id,
        }
        for row in rows
    ]


async def replay(pipeline: RetrievalPipeline, question: str) -> dict[str, list[str]]:
    """Run the pipeline's two searches and its fusion for one question.

    Args:
        pipeline: The loaded pipeline (no reranker needed).
        question: The question text.

    Returns:
        The doc ids of the gte top 50, the BM25 top 50 and the fused top 50, best first.
    """
    clean = normalize_text(question)
    semantic, keyword = await asyncio.gather(
        pipeline.semantic_search(clean), pipeline.keyword_search(clean)
    )
    fused = reciprocal_rank_fusion({"gte": semantic, "bm25": keyword}, RERANK_DEPTH)

    return {
        "gte": [hit.doc_id for hit in semantic],
        "bm25": [hit.doc_id for hit in keyword],
        "fused": [document.doc_id for document in fused],
    }


def origin(doc_id: str, rankings: dict[str, list[str]]) -> str:
    """Name the ranking(s) that found a document.

    Args:
        doc_id: The document.
        rankings: The replayed gte and BM25 top 50.

    Returns:
        "both", "gte only", "bm25 only" or "neither".
    """
    in_gte, in_bm25 = doc_id in rankings["gte"], doc_id in rankings["bm25"]
    if in_gte and in_bm25:
        return "both"
    if in_gte:
        return "gte only"

    return "bm25 only" if in_bm25 else "neither"


def summarise(records: list[dict[str, Any]]) -> None:
    """Print where the right documents came from, overall and per question type.

    Args:
        records: Questions with their replayed rankings.
    """
    answerable = [record for record in records if record["expected"]]
    in_candidates: Counter[str] = Counter()
    in_final: Counter[str] = Counter()
    correct_final: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = {}
    mismatched = 0
    for record in answerable:
        rankings = record["rankings"]
        mismatched += not set(record["final"]) <= set(rankings["fused"])
        for doc_id in record["expected"]:
            where = origin(doc_id, rankings)
            in_candidates[where] += 1
            by_type.setdefault(record["type"], Counter())[where] += 1
            if doc_id in record["final"]:
                in_final[where] += 1
                if record["correct"]:
                    correct_final[where] += 1

    total = sum(in_candidates.values())
    print(f"{len(answerable)} answerable questions, {total} right documents")
    print(f"final 5 not inside the replayed fused 50 (drift since the run): {mismatched}")
    for title, counts in (
        ("right documents in the top-50 lists", in_candidates),
        ("right documents that reached the final 5", in_final),
        ("... of those, in questions judged correct", correct_final),
    ):
        print(f"\n{title} ({sum(counts.values())}):")
        for where in ("both", "gte only", "bm25 only", "neither"):
            share = counts[where] / max(1, sum(counts.values()))
            print(f"  {where:10s} {counts[where]:4d}  {share:6.1%}")
    print("\nper type (both / gte only / bm25 only / neither):")
    for question_type, counts in sorted(by_type.items()):
        print(
            f"  {question_type:25s} {counts['both']:3d} / {counts['gte only']:3d} / "
            f"{counts['bm25 only']:3d} / {counts['neither']:3d}"
        )


async def main() -> None:
    """Replay every question not yet in the output file, then print the summary."""
    records = {record["id"]: record for record in load_questions()}
    done: dict[str, Any] = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    todo = [record for question_id, record in records.items() if question_id not in done]
    if todo:
        pipeline = RetrievalPipeline.load(rerank=False)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with RunProgress(
            get_engine(), "replay gte + BM25 rankings (free)", len(todo), unit="questions"
        ) as progress:
            for record in todo:
                done[record["id"]] = await replay(pipeline, record["text"])
                OUTPUT.write_text(json.dumps(done))
                progress.advance(record["type"], 1, note="local, $0")
    summarise(
        [{**record, "rankings": done[question_id]} for question_id, record in records.items()]
    )


if __name__ == "__main__":
    asyncio.run(main())
