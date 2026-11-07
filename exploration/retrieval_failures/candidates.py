"""Save the fused top 50 candidates of the retrieval pipeline for every semantic question.

These are the documents the reranker sees: it only reorders them, so they are fetched without
reranking. A document missing from them was never found by search.

Usage: uv run python exploration/retrieval_failures/candidates.py OUT.json
"""

import asyncio
import json
import sys
from pathlib import Path

from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.models import QuestionType
from train_rag.pipelines.retrieval import CANDIDATES, RetrievalPipeline
from train_rag.repository.question import load_questions


async def main() -> None:
    """Fetch every semantic question's candidates and write them to OUT."""
    with Session(get_engine()) as session:
        questions = load_questions(session, EnterpriseRagBenchLoader.name, QuestionType.SEMANTIC)
    pipeline = RetrievalPipeline.load(rerank=False)
    candidates = {}
    try:
        for index, question in enumerate(questions, 1):
            documents = await pipeline.run(question.text, CANDIDATES)
            candidates[question.id] = [document.doc_id for document in documents]
            print(f"{index}/{len(questions)}", flush=True)
    finally:
        await pipeline.close()
    Path(sys.argv[1]).write_text(json.dumps(candidates, indent=1))


asyncio.run(main())
