"""Pick benchmark questions to label for ambiguity, by how well their top 2 documents match.

Only questions with exactly one expected document are used: questions answered from
several documents, or from none, are a different problem. Each is searched with gte
(semantic search alone). Its score is the
similarity of its second-best document: when even the second document matches the question
closely, two documents may answer it. The 10 highest, 10 lowest and 10 middle scores are
written to sample.json, with each question's top 2 documents, for labelling in label.py.

Usage: uv run python exploration/question_ambiguity/sample.py
"""

import asyncio
import json
from pathlib import Path

from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.pipelines.retrieval import RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.repository.question import load_questions

OUT = Path(__file__).parent / "sample.json"
PER_GROUP = 10


async def main() -> None:
    """Score every question by its second-best document and write the sample."""
    with Session(get_engine()) as session:
        questions = [
            question
            for question in load_questions(session, EnterpriseRagBenchLoader.name)
            if len(question.expected_doc_ids) == 1
        ]
    pipeline = RetrievalPipeline.load(rerank=False)
    scored = []
    try:
        for index, question in enumerate(questions, 1):
            hits = (await pipeline.semantic_search(normalize_text(question.text)))[:2]
            scored.append(
                {
                    "question_id": question.id,
                    "question_type": question.question_type.value,
                    "score": 1 - hits[1].distance,
                    "top2": [
                        {"doc_id": hit.doc_id, "gte_similarity": 1 - hit.distance} for hit in hits
                    ],
                }
            )
            print(f"\r{index}/{len(questions)}", end="", flush=True)
    finally:
        await pipeline.close()
    print()

    scored.sort(key=lambda row: row["score"])
    middle = len(scored) // 2 - PER_GROUP // 2
    groups = {
        "low": scored[:PER_GROUP],
        "middle": scored[middle : middle + PER_GROUP],
        "high": scored[-PER_GROUP:],
    }
    sample = [{**row, "group": name} for name, rows in groups.items() for row in rows]
    OUT.write_text(json.dumps(sample, indent=1))
    for name, rows in groups.items():
        print(f"{name}: {rows[0]['score']:.3f} to {rows[-1]['score']:.3f}")


asyncio.run(main())
