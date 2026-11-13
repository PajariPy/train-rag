"""Search the whole corpus with the dev questions: base gte against the fine-tuned gte.

The dev check in train.py ranks each right document among its question's own ~51
candidates. This runs the real search over all 512k documents instead, once per embedder,
exactly as production does (gte top 50 + BM25 top 50, fused to 50, reranked to 10), and
records where each dev question's right document lands:

- in the embedder's own top 50 (what training changed);
- in the fused top 50 (what the reranker gets to see);
- in the reranker's top 10 (what search returns).

Results per embedder go to data/dev_search_<name>.json (doc rank per question; resumable),
then a table per question type is printed. Free and local: no LLM, about 1 h for both.

Usage:
    uv run python exploration/embedder_finetune/dev_search.py
"""

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from train_rag.database import get_engine
from train_rag.embedder import Embedder, FineTunedGteEmbedder, GteModernBertEmbedder
from train_rag.fusion import reciprocal_rank_fusion
from train_rag.pipelines.retrieval import RERANK_DEPTH, RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress

DATA = Path(__file__).parent / "data"
DEV = DATA / "dev_set.jsonl"
SAVE_EVERY = 20


def rank_of(doc_ids: list[str], positive: str) -> int | None:
    """The 1-based rank of the right document, or None if it isn't in the list.

    Args:
        doc_ids: Ranked doc ids, best first.
        positive: The right document.

    Returns:
        Its rank, or None.
    """
    return doc_ids.index(positive) + 1 if positive in doc_ids else None


async def search_one(pipeline: RetrievalPipeline, question: str, positive: str) -> dict[str, Any]:
    """Run one question through search and note the right document's rank at each stage.

    Args:
        pipeline: The loaded pipeline (with the reranker).
        question: The dev question.
        positive: Its right document.

    Returns:
        Ranks in the embedder's top 50, the fused top 50 and the reranked top 10.
    """
    # 1. Search both ways, as production does.
    clean = normalize_text(question)
    semantic, keyword = await asyncio.gather(
        pipeline.semantic_search(clean), pipeline.keyword_search(clean)
    )

    # 2. Fuse to the reranker's depth, then rerank.
    fused = reciprocal_rank_fusion({"gte": semantic, "bm25": keyword}, RERANK_DEPTH)
    documents = await pipeline.describe(fused)
    assert pipeline.reranker is not None
    reranked = await pipeline.rerank(pipeline.reranker, clean, documents)

    return {
        "embedder": rank_of([hit.doc_id for hit in semantic], positive),
        "fused": rank_of([document.doc_id for document in fused], positive),
        "reranked": rank_of([document.doc_id for document in reranked[:10]], positive),
    }


async def run_embedder(embedder: Embedder, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Search every dev question with one embedder, resuming from its saved results.

    Args:
        embedder: The embedder to search with.
        rows: Dev questions.

    Returns:
        Question id -> ranks.
    """
    path = DATA / f"dev_search_{embedder.name}.json"
    results: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    todo = [row for row in rows if row["id"] not in results]

    if not todo:
        return results

    pipeline = RetrievalPipeline.load(rerank=True, embedder=embedder)

    with RunProgress(
        get_engine(),
        f"dev search over the corpus: {embedder.name} (free)",
        len(todo),
        unit="questions",
    ) as progress:
        for count, row in enumerate(todo, 1):
            results[row["id"]] = await search_one(pipeline, row["question"], row["positive"])
            progress.advance(embedder.name, 1, note="local, $0")

            if count % SAVE_EVERY == 0:
                path.write_text(json.dumps(results))

    path.write_text(json.dumps(results))
    await pipeline.close()

    return results


def report(rows: list[dict[str, Any]], by_model: dict[str, dict[str, Any]]) -> None:
    """Print, per question type, how often the right document reaches each stage.

    Args:
        rows: Dev questions.
        by_model: Embedder name -> question id -> ranks.
    """
    types = sorted({row["question_type"] for row in rows})

    for stage, label in (
        ("embedder", "in the embedder's top 50"),
        ("fused", "in the fused top 50"),
        ("reranked", "in the reranked top 10"),
    ):
        print(f"\nRight document {label}:")
        print(f"  {'type':26s}" + "".join(f"{name:>22s}" for name in by_model))

        for question_type in ["all", *types]:
            chosen = [row for row in rows if question_type in ("all", row["question_type"])]
            cells = []

            for results in by_model.values():
                found = sum(results[row["id"]][stage] is not None for row in chosen)
                cells.append(f"{found:4d}/{len(chosen):<4d} {found / len(chosen):6.0%}")

            print(f"  {question_type:26s}" + "".join(f"{cell:>22s}" for cell in cells))

    ranks: dict[str, list[int]] = defaultdict(list)

    for name, results in by_model.items():
        for row in rows:
            rank = results[row["id"]]["reranked"]
            ranks[name].append(rank or 11)

    print("\nRight document first after reranking:")

    for name, values in ranks.items():
        print(f"  {name:22s} {sum(rank == 1 for rank in values) / len(values):6.0%}")


def main() -> None:
    """Search with both embedders and print the comparison."""
    rows = [json.loads(line) for line in DEV.read_text().splitlines() if line]
    by_model: dict[str, dict[str, Any]] = {}

    for embedder in (GteModernBertEmbedder(), FineTunedGteEmbedder()):
        by_model[embedder.name] = asyncio.run(run_embedder(embedder, rows))

    report(rows, by_model)


if __name__ == "__main__":
    main()
