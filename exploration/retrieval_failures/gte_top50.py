"""Save gte's own top 50 documents, with their similarities, for a few questions.

This is the semantic search half of the retrieval pipeline alone, before fusion with BM25.

Usage: uv run python exploration/retrieval_failures/gte_top50.py OUT.json QUESTION_ID...
"""

import asyncio
import json
import sys
from pathlib import Path

from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.pipelines.retrieval import RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.repository.question import load_questions


async def main() -> None:
    """Search every question with gte and write the results to OUT."""
    wanted = set(sys.argv[2:])
    with Session(get_engine()) as session:
        questions = [
            question
            for question in load_questions(session, EnterpriseRagBenchLoader.name)
            if question.id in wanted
        ]
    pipeline = RetrievalPipeline.load(rerank=False)
    results = {}
    try:
        for question in questions:
            hits = await pipeline.semantic_search(normalize_text(question.text))
            results[question.id] = [
                {"doc_id": hit.doc_id, "gte_similarity": 1 - hit.distance} for hit in hits
            ]
            print(question.id, len(hits), flush=True)
    finally:
        await pipeline.close()
    Path(sys.argv[1]).write_text(json.dumps(results, indent=1))


asyncio.run(main())
