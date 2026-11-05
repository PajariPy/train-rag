"""Would shorter reranker inputs be faster without losing the right document?

For every semantic question the candidates are the pipeline's fused top 50 (what the
reranker sees today) plus the expected document if it is missing. Each candidate is
reranked three ways:

- chunk: today's input, the title and the document's best chunk (up to 24,000 characters);
- window: the title and the WINDOW_CHARS window holding the most query words (as BM25
  matches them: stemmed, no stopwords);
- start: the title and the first WINDOW_CHARS characters.

For each way: the reranking time per document, and the expected document's rank among the
candidates. Questions whose expected document is in the fused top 50 show what the change
would do to today's pipeline; the others show whether the reranker would still pick it.

Usage: uv run python exploration/retrieval_failures/rerank_window.py OUT.json
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import torch
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.evaluation.snippets import STEMMER, WORD, query_terms
from train_rag.fusion import reciprocal_rank_fusion
from train_rag.models import QuestionType
from train_rag.pipelines.retrieval import RERANK_DEPTH, RetrievalPipeline
from train_rag.processing.normalize import normalize_content, normalize_text
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.repository.question import load_questions

WINDOW_CHARS = 2_000
STEP = 250
CHUNK_CHARS = 24_000  # the gte chunk size: today's input for a document outside the top 50


def best_window(text: str, terms: set[str]) -> str:
    """The WINDOW_CHARS window holding the most query words (start of text if none match)."""
    hits = [
        word.start()
        for word in WORD.finditer(text)
        if STEMMER.stemWord(word.group().lower()) in terms
    ]
    if not hits:
        return text[:WINDOW_CHARS]
    best_start, best_count, low = 0, -1, 0
    for start in range(0, max(1, len(text) - WINDOW_CHARS + STEP), STEP):
        while low < len(hits) and hits[low] < start:
            low += 1
        high = low
        while high < len(hits) and hits[high] < start + WINDOW_CHARS:
            high += 1
        if high - low > best_count:
            best_start, best_count = start, high - low

    return text[best_start : best_start + WINDOW_CHARS]


def start_of(text: str) -> str:
    """The title and the first WINDOW_CHARS characters after it."""
    title, _, body = text.partition("\n\n")

    return f"{title}\n\n{body[:WINDOW_CHARS]}"


def timed_scores(reranker, query: str, texts: list[str]) -> tuple[list[float], float]:  # noqa: ANN001
    """Rerank texts and time it on the GPU."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    scores = reranker.score(query, texts)
    torch.cuda.synchronize()

    return scores, time.perf_counter() - start


def rank(scores: dict[str, float], target: str) -> int:
    """Rank of the target among all candidates (1 = top)."""
    return 1 + sum(score > scores[target] for doc_id, score in scores.items() if doc_id != target)


async def main() -> None:
    """Rerank every question's candidates three ways and write the ranks and times."""
    dataset = EnterpriseRagBenchLoader.name
    with Session(get_engine()) as session:
        questions = load_questions(session, dataset, QuestionType.SEMANTIC)
    pipeline = RetrievalPipeline.load(rerank=True)
    reranker = pipeline.reranker
    assert reranker is not None
    reranker.score("warm up", ["warm up text"])
    rows = []
    try:
        for number, question in enumerate(questions, 1):
            clean = normalize_text(question.text)
            semantic, keyword = await asyncio.gather(
                pipeline.semantic_search(clean), pipeline.keyword_search(clean)
            )
            fused = reciprocal_rank_fusion({"gte": semantic, "bm25": keyword}, RERANK_DEPTH)
            described = await pipeline.describe(fused)
            expected = question.expected_doc_ids[0]
            ids = [doc.doc_id for doc in described]
            in_top50 = expected in ids
            with Session(get_engine()) as session:
                rows_ids = ids_of_doc_ids(session, dataset, [*ids, expected])
                full = {
                    d.doc_id: f"{d.title}\n\n{normalize_content(d.content)}"
                    for d in load_documents(session, sorted(rows_ids))
                }
            chunk = {doc.doc_id: doc.text for doc in described}
            if not in_top50 and expected in full:
                chunk[expected] = full[expected][:CHUNK_CHARS]
                ids.append(expected)
            terms = query_terms(clean)
            inputs = {
                "chunk": [chunk[d] for d in ids],
                "window": [
                    full[d].split("\n", 1)[0] + "\n\n" + best_window(full[d], terms) for d in ids
                ],
                "start": [start_of(full[d]) for d in ids],
            }
            row: dict[str, object] = {"question_id": question.id, "in_top50": in_top50}
            for way, texts in inputs.items():
                scores, seconds = timed_scores(reranker, clean, texts)
                by_id = dict(zip(ids, scores, strict=True))
                row[way] = {
                    "seconds": seconds,
                    "documents": len(texts),
                    "chars": sum(len(text) for text in texts),
                    "rank": rank(by_id, expected) if expected in by_id else None,
                }
            rows.append(row)
            if number % 25 == 0:
                print(f"{number}/{len(questions)}", flush=True)
    finally:
        await pipeline.close()
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
    print(f"wrote {sys.argv[1]}", flush=True)


asyncio.run(main())
