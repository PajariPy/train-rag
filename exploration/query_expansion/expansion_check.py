"""HyDE and query rewriting on the dev set, on top of the union rerank.

Production search (fine-tuned gte + BM25, top 50 each, fused to 50, reranked) misses the
right document for many semantic questions because it never reaches the reranker's list.
Reranking the whole union of both top-50 lists helps (15 Nov); this adds more searches:

- HyDE: Gemini writes a passage that would answer the question; gte and BM25 search with it.
- Query rewriting: Gemini rewrites the question into 4 search queries; gte and BM25 search
  with each.

Every candidate any search returns is scored once by the reranker against the original
question. A cross-encoder scores each document on its own, so the ranking of any candidate
set (fused 50, union, union + HyDE, ...) is read from the same scores without reranking again.

Two phases, both resumable:
1. All 422 dev questions, question searches only (free): fused 50 vs union.
2. The 118 semantic dev questions, plus HyDE and rewrites (Gemini, ~$0.40).

Writes data/expansion_check.json (per question: candidate lists and reranker scores) and
data/expansions.json (Gemini's passages and queries), then prints the summary.

Usage:
    uv run python exploration/query_expansion/expansion_check.py
"""

import asyncio
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from train_rag.database import get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.evaluation.pipeline_answer import GEMINI_MODEL
from train_rag.evaluation.usage import UsageMeter
from train_rag.fusion import reciprocal_rank_fusion
from train_rag.models.retrieval import RetrievedDocument, SearchHit
from train_rag.pipelines.retrieval import CANDIDATES, RERANK_DEPTH, RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress

DATA = Path(__file__).parent / "data"
ENV_FILE = Path(__file__).parents[2] / ".env"
DEV = Path(__file__).parents[1] / "embedder_finetune/data/dev_set.jsonl"
OUTPUT = DATA / "expansion_check.json"
EXPANSIONS = DATA / "expansions.json"
EXPANDED_TYPE = "semantic"
REWRITES = 4
WORKERS = 8

HYDE_INSTRUCTIONS = (
    "You write a passage from a company's internal documents (a Slack thread, an email, a "
    "Jira ticket, a Google Drive document, a meeting transcript, a CRM record...) that "
    "answers the question. Write it the way that document itself would read: concrete, "
    "120-200 words, in the vocabulary such a document would use. Invent plausible details "
    "where you don't know them. Return only the passage."
)
REWRITE_INSTRUCTIONS = (
    f"Rewrite the question into {REWRITES} search queries for a search engine over a "
    "company's internal documents (Slack, email, Jira, Google Drive, meeting transcripts, "
    "CRM). The first is a keyword query: the key terms plus the synonyms the document may "
    "use instead. The others describe, in different words from the question, what the "
    "document that answers it would say. Don't answer the question."
)


class Rewrites(BaseModel):
    """Gemini's search queries for one question."""

    queries: list[str]


def generate(client: genai.Client, usage: UsageMeter, question: str) -> dict[str, Any]:
    """Ask Gemini for a HyDE passage and the rewritten queries of one question.

    Args:
        client: Gemini client.
        usage: Counts the tokens of both calls.
        question: The question text.

    Returns:
        The passage and the queries.
    """
    thinking = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    passage = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Question: {question}",
        config=types.GenerateContentConfig(
            system_instruction=HYDE_INSTRUCTIONS, thinking_config=thinking
        ),
    )
    usage.add(passage)
    rewritten = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Question: {question}",
        config=types.GenerateContentConfig(
            system_instruction=REWRITE_INSTRUCTIONS,
            response_mime_type="application/json",
            response_schema=Rewrites,
            thinking_config=thinking,
        ),
    )
    usage.add(rewritten)
    parsed = rewritten.parsed
    queries = parsed.queries if isinstance(parsed, Rewrites) else []

    return {"hyde": (passage.text or "").strip(), "rewrites": queries[:REWRITES]}


def load_expansions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate (or load) the HyDE passage and rewrites of every semantic dev question.

    Args:
        rows: Semantic dev questions.

    Returns:
        Question id -> passage and queries.
    """
    done: dict[str, Any] = json.loads(EXPANSIONS.read_text()) if EXPANSIONS.exists() else {}
    todo = [row for row in rows if row["id"] not in done]
    if not todo:
        return done

    load_dotenv(ENV_FILE)
    client, usage = genai.Client(), UsageMeter()
    title = "query expansion: Gemini HyDE + rewrites (dev semantic)"
    with (
        RunProgress(get_engine(), title, len(todo), unit="questions") as progress,
        ThreadPoolExecutor(WORKERS) as pool,
    ):
        futures = {row["id"]: pool.submit(generate, client, usage, row["question"]) for row in todo}
        for question_id, future in futures.items():
            done[question_id] = future.result()
            EXPANSIONS.write_text(json.dumps(done, indent=1))
            progress.advance("generating", 1, note=f"Gemini: ${usage.cost():.2f}")
    print(usage.report())

    return done


async def search(pipeline: RetrievalPipeline, text: str) -> tuple[list[SearchHit], list[SearchHit]]:
    """Run gte and BM25 for one search text.

    Args:
        pipeline: The loaded pipeline.
        text: The question, passage or query.

    Returns:
        The gte top 50 and the BM25 top 50, best first.
    """
    clean = normalize_text(text)
    semantic, keyword = await asyncio.gather(
        pipeline.semantic_search(clean), pipeline.keyword_search(clean)
    )

    return semantic, keyword


async def replay(
    pipeline: RetrievalPipeline, row: dict[str, Any], expansion: dict[str, Any] | None
) -> dict[str, Any]:
    """Run every search for one question and score all candidates once.

    Args:
        pipeline: The loaded pipeline with its reranker.
        row: The dev question.
        expansion: Its HyDE passage and rewrites, or None for question searches only.

    Returns:
        The doc ids of each search ("gte:q", "bm25:q", "gte:hyde", "bm25:rw1", ...), the
        fused 50 of the question searches, the reranker score of every candidate and the
        seconds the reranking took.
    """
    texts = {"q": row["question"]}
    if expansion is not None:
        texts["hyde"] = expansion["hyde"]
        texts |= {f"rw{i}": query for i, query in enumerate(expansion["rewrites"], 1)}

    lists: dict[str, list[SearchHit]] = {}
    for name, text in texts.items():
        lists[f"gte:{name}"], lists[f"bm25:{name}"] = await search(pipeline, text)
    fused = reciprocal_rank_fusion({"gte": lists["gte:q"], "bm25": lists["bm25:q"]}, RERANK_DEPTH)

    # Every candidate once, described like the pipeline does (title + best chunk).
    every = reciprocal_rank_fusion(lists, CANDIDATES * len(lists))
    documents: list[RetrievedDocument] = await pipeline.describe(every)
    assert pipeline.reranker is not None
    started = time.perf_counter()
    reranked = await pipeline.rerank(pipeline.reranker, normalize_text(row["question"]), documents)
    seconds = time.perf_counter() - started

    return {
        "lists": {name: [hit.doc_id for hit in hits] for name, hits in lists.items()},
        "fused": [document.doc_id for document in fused],
        "scores": {document.doc_id: document.rerank_score for document in reranked},
        "rerank_seconds": seconds,
    }


def candidates(record: dict[str, Any], names: list[str]) -> set[str]:
    """The union of some search lists.

    Args:
        record: One replayed question.
        names: Search list names.

    Returns:
        Their doc ids.
    """
    return {doc_id for name in names if name in record["lists"] for doc_id in record["lists"][name]}


def ranked(record: dict[str, Any], pool: set[str]) -> list[str]:
    """Order a candidate set by the reranker's scores.

    Args:
        record: One replayed question.
        pool: Candidate doc ids.

    Returns:
        The doc ids, best first.
    """
    return sorted(pool, key=lambda doc_id: record["scores"][doc_id], reverse=True)


STRATEGIES: dict[str, Callable[[dict[str, Any]], set[str]]] = {
    "fused 50 (production)": lambda r: set(r["fused"]),
    "union": lambda r: candidates(r, ["gte:q", "bm25:q"]),
    "union + HyDE gte": lambda r: candidates(r, ["gte:q", "bm25:q", "gte:hyde"]),
    "union + HyDE gte+bm25": lambda r: candidates(r, ["gte:q", "bm25:q", "gte:hyde", "bm25:hyde"]),
    "union + rewrites": lambda r: candidates(
        r, ["gte:q", "bm25:q"] + [f"{s}:rw{i}" for s in ("gte", "bm25") for i in range(1, 5)]
    ),
    "union + HyDE + rewrites": lambda r: set(r["scores"]),
}


def summarise(rows: list[dict[str, Any]], records: dict[str, Any]) -> None:
    """Print the dev metrics of every candidate strategy.

    Args:
        rows: Dev questions.
        records: Question id -> replayed searches and scores.
    """
    ms_per_doc = (
        1000
        * sum(r["rerank_seconds"] for r in records.values())
        / sum(len(r["scores"]) for r in records.values())
    )
    groups = {"all dev": rows} | {
        kind: [row for row in rows if row["question_type"] == kind]
        for kind in sorted({row["question_type"] for row in rows})
    }
    for group, chosen in groups.items():
        expanded = all("gte:hyde" in records[row["id"]]["lists"] for row in chosen)
        print(f"\n{group} ({len(chosen)} questions)")
        print(
            f"  {'strategy':24s} {'in list':>8s} {'top 1':>7s} {'top 5':>7s} {'top 10':>7s}"
            f" {'docs':>6s} {'rerank s':>9s}"
        )
        for name, strategy in STRATEGIES.items():
            if "HyDE" in name or "rewrites" in name:
                if not expanded:
                    continue
            found = {1: 0, 5: 0, 10: 0}
            in_list = sizes = 0
            for row in chosen:
                record = records[row["id"]]
                pool = strategy(record)
                order = ranked(record, pool)
                sizes += len(pool)
                in_list += row["positive"] in pool
                for k in found:
                    found[k] += row["positive"] in order[:k]
            n = len(chosen)
            print(
                f"  {name:24s} {in_list / n:8.1%} {found[1] / n:7.1%} {found[5] / n:7.1%}"
                f" {found[10] / n:7.1%} {sizes / n:6.0f} {sizes / n * ms_per_doc / 1000:9.1f}"
            )
    print(f"\nrerank speed: {ms_per_doc:.0f} ms per document")


async def main() -> None:
    """Replay the dev questions (expanded for semantic ones), then print the summary."""
    rows = [json.loads(line) for line in DEV.read_text().splitlines()]
    expansions = load_expansions([row for row in rows if row["question_type"] == EXPANDED_TYPE])
    records: dict[str, Any] = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    todo = [row for row in rows if row["id"] not in records]
    print(f"{len(records)} questions already replayed, {len(todo)} to go")
    if todo:
        pipeline = RetrievalPipeline.load(rerank=True, embedder=FineTunedGteEmbedder())
        title = "query expansion: dev searches + rerank (local)"
        with RunProgress(get_engine(), title, len(todo), unit="questions") as progress:
            for row in todo:
                records[row["id"]] = await replay(pipeline, row, expansions.get(row["id"]))
                OUTPUT.write_text(json.dumps(records))
                progress.advance(row["question_type"], 1, note="local, $0")
        await pipeline.close()
    summarise(rows, records)


if __name__ == "__main__":
    asyncio.run(main())
