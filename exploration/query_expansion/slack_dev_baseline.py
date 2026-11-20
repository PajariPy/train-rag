"""Today's best search on the Slack semantic dev questions (`dev_slack`): the baseline to beat.

Best setup (16 Nov decision): fine-tuned gte top 50 on the question, BM25 top 50 on the
question with 4 Gemini rewrites appended, the union of both lists reranked against the
question. Plain BM25 on the question is kept too, for comparison.

The 200 questions were written from Slack threads (training_questions/slack_dev_sources.py,
then generate.py); each one's right document is the thread it was written from. Gemini
writes only the rewrites here (no HyDE passage), with the same prompt as on the benchmark.

Stores every list and the reranker score of every candidate, so the mined-field test can
filter or boost the same candidates without searching again. Resumable.

Writes data/slack_dev_expansions.json and data/slack_dev_baseline.json, then prints the
summary.

Usage:
    uv run python exploration/query_expansion/slack_dev_baseline.py
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from expansion_check import (
    DATA,
    ENV_FILE,
    GEMINI_MODEL,
    REWRITE_INSTRUCTIONS,
    REWRITES,
    WORKERS,
    Rewrites,
)
from expansion_merge import vector_search
from google import genai
from google.genai import types

from train_rag.database import get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.evaluation.usage import UsageMeter
from train_rag.fusion import reciprocal_rank_fusion
from train_rag.pipelines.retrieval import CANDIDATES, RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress

QUESTIONS = Path(__file__).parents[1] / "training_questions" / "data" / "questions.jsonl"
SPLIT = "dev_slack"
EXPANSIONS = DATA / "slack_dev_expansions.json"
OUTPUT = DATA / "slack_dev_baseline.json"
SETUPS = [("q", "q"), ("q", "concat")]  # (gte query, BM25 query)


def load_questions() -> list[dict[str, Any]]:
    """The dev_slack questions: id (the source thread), text, right documents."""
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line]

    return [
        {"id": row["doc_id"], "text": row["question"], "expected": [row["doc_id"]]}
        for row in rows
        if row["split"] == SPLIT and row["valid"]
    ]


def rewrite(client: genai.Client, usage: UsageMeter, question: str) -> list[str]:
    """Ask Gemini for the rewritten search queries of one question."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Question: {question}",
        config=types.GenerateContentConfig(
            system_instruction=REWRITE_INSTRUCTIONS,
            response_mime_type="application/json",
            response_schema=Rewrites,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        ),
    )
    usage.add(response)
    parsed = response.parsed

    return (parsed.queries if isinstance(parsed, Rewrites) else [])[:REWRITES]


def load_rewrites(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Generate (or load) the rewrites of every question."""
    done: dict[str, list[str]] = json.loads(EXPANSIONS.read_text()) if EXPANSIONS.exists() else {}
    todo = [record for record in records if record["id"] not in done]
    if not todo:
        return done

    load_dotenv(ENV_FILE)
    client, usage = genai.Client(), UsageMeter()
    title = "dev_slack baseline: Gemini rewrites"
    with (
        RunProgress(get_engine(), title, len(todo), unit="questions") as progress,
        ThreadPoolExecutor(WORKERS) as pool,
    ):
        futures = {
            record["id"]: pool.submit(rewrite, client, usage, record["text"]) for record in todo
        }
        for question_id, future in futures.items():
            done[question_id] = future.result()
            EXPANSIONS.write_text(json.dumps(done, indent=1))
            progress.advance("rewriting", 1, note=f"Gemini: ${usage.cost():.3f}")
    print(usage.report())

    return done


async def replay(pipeline: RetrievalPipeline, question: str, rewrites: list[str]) -> dict[str, Any]:
    """Run the searches for one question and score every candidate once against the question.

    Returns:
        The doc ids of each search ("gte:q", "bm25:q", "bm25:concat") and the reranker score
        of every candidate.
    """
    clean = normalize_text(question)
    concat = normalize_text(" ".join([question, *rewrites]))
    vector = pipeline.embedder.embed([clean])[0]
    lists = {
        "gte:q": await vector_search(pipeline, vector),
        "bm25:q": await pipeline.keyword_search(clean),
        "bm25:concat": await pipeline.keyword_search(concat),
    }
    every = reciprocal_rank_fusion(lists, CANDIDATES * len(lists))
    described = await pipeline.describe(every)
    assert pipeline.reranker is not None
    reranked = await pipeline.rerank(pipeline.reranker, clean, described)

    return {
        "lists": {name: [hit.doc_id for hit in hits] for name, hits in lists.items()},
        "scores": {document.doc_id: document.rerank_score or 0.0 for document in reranked},
    }


def summarise(records: list[dict[str, Any]], results: dict[str, Any]) -> None:
    """Print each setup's result: right document in each list, in the union, union top k."""
    n = len(records)
    print(f"dev_slack semantic ({n} questions), fine-tuned gte, union of both top 50 reranked")
    for name in ("gte:q", "bm25:q", "bm25:concat"):
        hits = sum(bool(set(r["expected"]) & set(results[r["id"]]["lists"][name])) for r in records)
        print(f"  {name:12s} right document in its top 50: {hits / n:.1%}")
    print(f"  {'gte':4s} {'BM25':7s} {'in union':>9s} {'top 1':>7s} {'top 5':>7s} {'top 10':>7s} {'docs':>5s}")
    for gte, bm25 in SETUPS:
        found = size = 0
        top = {1: 0, 5: 0, 10: 0}
        for record in records:
            result = results[record["id"]]
            pool = set(result["lists"][f"gte:{gte}"]) | set(result["lists"][f"bm25:{bm25}"])
            order = sorted(pool, key=lambda doc_id: result["scores"][doc_id], reverse=True)
            expected = set(record["expected"])
            size += len(pool)
            found += bool(expected & pool)
            for k in top:
                top[k] += bool(expected & set(order[:k]))
        print(
            f"  {gte:4s} {bm25:7s} {found / n:9.1%} {top[1] / n:7.1%} {top[5] / n:7.1%}"
            f" {top[10] / n:7.1%} {size / n:5.0f}"
        )


async def main() -> None:
    """Rewrite and replay every dev_slack question, then print the summary."""
    records = load_questions()
    rewrites = load_rewrites(records)
    results: dict[str, Any] = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    todo = [record for record in records if record["id"] not in results]
    print(f"{len(results)} questions already replayed, {len(todo)} to go", flush=True)
    if todo:
        pipeline = RetrievalPipeline.load(rerank=True, embedder=FineTunedGteEmbedder())
        title = "dev_slack baseline: search + rerank (local)"
        with RunProgress(get_engine(), title, len(todo), unit="questions") as progress:
            for record in todo:
                results[record["id"]] = await replay(
                    pipeline, record["text"], rewrites[record["id"]]
                )
                OUTPUT.write_text(json.dumps(results))
                progress.advance("searching", 1, note="local, $0")
        await pipeline.close()
    summarise(records, results)


if __name__ == "__main__":
    asyncio.run(main())
