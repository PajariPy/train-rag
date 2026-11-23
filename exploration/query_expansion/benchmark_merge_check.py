"""The merged-query search (expansion_merge.py) on the benchmark's 125 semantic questions.

On dev, merging Gemini's expansions into the query before the search lifted semantic top 5
from 61.0% (union) to 66.9%, with the reranker still reading about 96 documents. Dev
questions are written like our training ones, so this checks the same setups on the
benchmark's own semantic questions, with the fine-tuned gte.

Same Gemini prompts as on dev (a HyDE passage and 4 rewritten queries, ~$0.25). Every
candidate is scored once by the reranker against the original question, so every setup is
ranked from the same scores. Resumable.

Writes data/benchmark_expansions.json and data/benchmark_merge_check.json, then prints the
summary.

Usage:
    uv run python exploration/query_expansion/benchmark_merge_check.py
"""

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from expansion_check import DATA, ENV_FILE, WORKERS, generate
from expansion_merge import MEAN_VARIANTS, fuse_ids, mean_vector, vector_search
from google import genai

from train_rag.database import get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.evaluation.usage import UsageMeter
from train_rag.fusion import reciprocal_rank_fusion
from train_rag.models.retrieval import SearchHit
from train_rag.pipelines.retrieval import CANDIDATES, RERANK_DEPTH, RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress

sys.path.append(str(Path(__file__).parents[1] / "embedder_finetune"))
from benchmark_semantic_check import load_questions  # noqa: E402

EXPANSIONS = DATA / "benchmark_expansions.json"
OUTPUT = DATA / "benchmark_merge_check.json"
# (gte list, BM25 list) pairs, each one gte search + one BM25 search.
SETUPS = [
    ("q", "q"),
    ("q", "rw1"),
    ("q", "concat"),
    ("mean_rw", "concat"),
    ("mean_rw_q50", "rw1"),
    ("mean_hyde", "q"),
    ("mean_hyde", "rw1"),
    ("mean_hyde", "concat"),
    ("mean_all", "rw1"),
    ("concat", "rw1"),
]


def load_expansions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate (or load) the HyDE passage and rewrites of every benchmark semantic question.

    Args:
        records: Benchmark semantic questions.

    Returns:
        Question id -> passage and queries.
    """
    done: dict[str, Any] = json.loads(EXPANSIONS.read_text()) if EXPANSIONS.exists() else {}
    todo = [record for record in records if record["id"] not in done]
    if not todo:
        return done

    load_dotenv(ENV_FILE)
    client, usage = genai.Client(), UsageMeter()
    title = "query expansion: Gemini HyDE + rewrites (benchmark semantic)"
    with (
        RunProgress(get_engine(), title, len(todo), unit="questions") as progress,
        ThreadPoolExecutor(WORKERS) as pool,
    ):
        futures = {
            record["id"]: pool.submit(generate, client, usage, record["text"]) for record in todo
        }
        for question_id, future in futures.items():
            done[question_id] = future.result()
            EXPANSIONS.write_text(json.dumps(done, indent=1))
            progress.advance("generating", 1, note=f"Gemini: ${usage.cost():.2f}")
    print(usage.report())

    return done


async def replay(
    pipeline: RetrievalPipeline, question: str, expansion: dict[str, Any]
) -> dict[str, Any]:
    """Run every search the setups need for one question and score all candidates once.

    Args:
        pipeline: The loaded pipeline with its reranker.
        question: The question text.
        expansion: Its HyDE passage and rewrites.

    Returns:
        The doc ids of each search ("gte:q", "bm25:rw1", "gte:mean_hyde", ...) and the
        reranker score of every candidate.
    """
    texts = {"q": question, "hyde": expansion["hyde"]}
    texts |= {f"rw{i}": query for i, query in enumerate(expansion["rewrites"], 1)}
    texts["concat"] = " ".join([question, *expansion["rewrites"]])
    names = list(texts)
    embedded = pipeline.embedder.embed([normalize_text(texts[name]) for name in names])
    vectors = dict(zip(names, embedded, strict=True))

    lists: dict[str, list[SearchHit]] = {
        "gte:q": await vector_search(pipeline, vectors["q"]),
        "gte:concat": await vector_search(pipeline, vectors["concat"]),
    }
    for name, (weight, parts) in MEAN_VARIANTS.items():
        lists[f"gte:{name}"] = await vector_search(pipeline, mean_vector(vectors, weight, parts))
    for name in ("q", "rw1", "concat"):
        lists[f"bm25:{name}"] = await pipeline.keyword_search(normalize_text(texts[name]))

    every = reciprocal_rank_fusion(lists, CANDIDATES * len(lists))
    described = await pipeline.describe(every)
    assert pipeline.reranker is not None
    reranked = await pipeline.rerank(pipeline.reranker, normalize_text(question), described)

    return {
        "lists": {name: [hit.doc_id for hit in hits] for name, hits in lists.items()},
        "scores": {document.doc_id: document.rerank_score or 0.0 for document in reranked},
    }


def summarise(records: list[dict[str, Any]], results: dict[str, Any]) -> None:
    """Print every setup's result on the benchmark's semantic questions.

    Args:
        records: Benchmark semantic questions with their right documents.
        results: Question id -> lists and scores.
    """
    n = len(records)
    print(f"benchmark semantic ({n} questions), fine-tuned gte, one gte + one BM25 search")
    print(
        f"  {'gte query':12s} {'BM25 query':10s} {'in union':>9s} {'union top 5':>12s}"
        f" {'docs':>5s} {'fused 50 top 5':>15s}"
    )
    for gte, bm25 in SETUPS:
        found = top5 = fused5 = size = 0
        for record in records:
            result = results[record["id"]]
            lists, scores = result["lists"], result["scores"]
            pool = set(lists[f"gte:{gte}"]) | set(lists[f"bm25:{bm25}"])
            order = sorted(pool, key=lambda doc_id: scores[doc_id], reverse=True)
            fused = fuse_ids([lists[f"gte:{gte}"], lists[f"bm25:{bm25}"]])[:RERANK_DEPTH]
            fused_order = sorted(fused, key=lambda doc_id: scores[doc_id], reverse=True)
            expected = set(record["expected"])
            size += len(pool)
            found += bool(expected & pool)
            top5 += bool(expected & set(order[:5]))
            fused5 += bool(expected & set(fused_order[:5]))
        print(
            f"  {gte:12s} {bm25:10s} {found / n:9.1%} {top5 / n:12.1%}"
            f" {size / n:5.0f} {fused5 / n:15.1%}"
        )


async def main() -> None:
    """Expand and replay every benchmark semantic question, then print the summary."""
    records = load_questions()
    expansions = load_expansions(records)
    results: dict[str, Any] = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    todo = [record for record in records if record["id"] not in results]
    print(f"{len(results)} questions already replayed, {len(todo)} to go")
    if todo:
        pipeline = RetrievalPipeline.load(rerank=True, embedder=FineTunedGteEmbedder())
        title = "query expansion: merged queries (benchmark semantic, local)"
        with RunProgress(get_engine(), title, len(todo), unit="questions") as progress:
            for record in todo:
                results[record["id"]] = await replay(
                    pipeline, record["text"], expansions[record["id"]]
                )
                OUTPUT.write_text(json.dumps(results))
                progress.advance("searching", 1, note="local, $0")
        await pipeline.close()
    summarise(records, results)


if __name__ == "__main__":
    asyncio.run(main())
