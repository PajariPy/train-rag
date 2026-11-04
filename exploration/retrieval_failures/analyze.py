"""Measure why the retrieval pipeline misses expected documents on semantic questions.

For each semantic question that run 8 (the plain pipeline) got wrong and whose expected
document was nowhere in the fused top 50, record:

- the expected document's exact BM25 rank and score, and its exact gte rank and cosine
  similarity, both over the whole corpus;
- the reranker's score for it;
- the five documents the pipeline picked instead, with their ranks and scores;
- whether the agent (run 14) found it.

Usage: uv run python exploration/retrieval_failures/analyze.py OUT.json QUESTION_ID...
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import text
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.index_bm25 import tokenize
from train_rag.pipelines.data_ingestion import FIELD
from train_rag.pipelines.retrieval import RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.repository.question import load_questions
from train_rag.tables import EMBEDDING_TABLES

PIPELINE_RUN = 8
AGENT_RUN = 14
OUT = Path(sys.argv[1])
QUESTION_IDS = sys.argv[2:]


def best_similarity(session: Session, literal: str, document_id: int, source: str) -> float:
    """Cosine similarity of a document's best chunk to the query vector."""
    distance = session.execute(
        text(
            f"select min(embedding <=> '{literal}') from {source}_embeddings_768 "
            "where document_id = :id and model = 'gte-modernbert-base' and field = :field"
        ),
        {"id": document_id, "field": FIELD},
    ).scalar_one()
    return 1 - float(distance)


def gte_rank(
    session: Session, vector: list[float], document_id: int, source: str
) -> tuple[int, float]:
    """Exact rank of a document by its best chunk's cosine distance, over every source."""
    literal = "[" + ",".join(f"{x:.7f}" for x in vector) + "]"
    own = session.execute(
        text(
            f"select min(embedding <=> '{literal}') from {source}_embeddings_768 "
            "where document_id = :id and model = 'gte-modernbert-base' and field = :field"
        ),
        {"id": document_id, "field": FIELD},
    ).scalar_one()
    closer = 0
    for table in EMBEDDING_TABLES[768].values():
        closer += session.execute(
            text(
                f"select count(*) from (select document_id, min(embedding <=> '{literal}') d "
                f"from {table.__tablename__} where model = 'gte-modernbert-base' "
                f"and field = :field group by document_id) best where d < :own"
            ),
            {"field": FIELD, "own": own},
        ).scalar_one()
    return closer + 1, 1 - float(own)


async def main() -> None:
    """Measure every question and write the records to OUT."""
    engine = get_engine()
    pipeline = RetrievalPipeline.load()
    bm25 = pipeline.bm25
    position = {doc.doc_id: index for index, doc in enumerate(bm25.documents)}
    records = []
    try:
        with Session(engine) as session:
            loaded = load_questions(session, EnterpriseRagBenchLoader.name, None, None)
            questions = {q.id: q for q in loaded}
            agent = dict(
                session.execute(
                    text("select question_id, trace from eval_results where run_id = :run"),
                    {"run": AGENT_RUN},
                ).all()
            )
            for qid in QUESTION_IDS:
                question = questions[qid]
                clean = normalize_text(question.text)
                scores = bm25.model.get_scores(tokenize([clean], bm25.stemmer)[0])
                vector = pipeline.embedder.embed([clean])[0]
                literal = "[" + ",".join(f"{x:.7f}" for x in vector) + "]"
                picked = await pipeline.run(question.text, 5)
                top_bm25 = float(np.max(scores))
                expected = []
                for doc_id in question.expected_doc_ids:
                    row = session.execute(
                        text("select id, source_type, title from documents where doc_id = :d"),
                        {"d": doc_id},
                    ).first()
                    if row is None:
                        continue
                    document_id, source, title = row
                    chunk = session.execute(
                        text(
                            f"select text from {source}_embeddings_768 where document_id = :id "
                            "and model = 'gte-modernbert-base' and field = :field "
                            "order by position limit 1"
                        ),
                        {"id": document_id, "field": FIELD},
                    ).scalar_one()
                    own_bm25 = float(scores[position[doc_id]])
                    rank, similarity = gte_rank(session, vector, document_id, source)
                    reranker = pipeline.reranker
                    rerank = reranker.score(clean, [chunk])[0] if reranker else None
                    agent_found = any(
                        doc_id in step.get("doc_ids", [])
                        for step in agent.get(qid, [])
                        if step["kind"] == "tool" and not step["arguments"].get("automatic")
                    )
                    expected.append(
                        {
                            "doc_id": doc_id,
                            "source": source,
                            "title": title.split("\n")[0][:120],
                            "text": chunk[:1500],
                            "chars": len(chunk),
                            "bm25_rank": int((scores > own_bm25).sum()) + 1,
                            "bm25_score": own_bm25,
                            "gte_rank": rank,
                            "gte_similarity": similarity,
                            "rerank_score": rerank,
                            "agent_found": agent_found,
                        }
                    )
                records.append(
                    {
                        "question_id": qid,
                        "question": question.text,
                        "gold_answer": question.gold_answer,
                        "top_bm25_score": top_bm25,
                        "expected": expected,
                        "picked": [
                            {
                                "doc_id": d.doc_id,
                                "source": d.source_type.value,
                                "title": d.title.split("\n")[0][:120],
                                "ranks": d.ranks,
                                "rerank_score": d.rerank_score,
                                "bm25_score": float(scores[position[d.doc_id]]),
                                "gte_similarity": best_similarity(
                                    session, literal, d.document_id, d.source_type.value
                                ),
                                "text": d.text[:400],
                            }
                            for d in picked
                        ],
                    }
                )
                print(f"{len(records)}/{len(QUESTION_IDS)} {qid}", flush=True)
    finally:
        await pipeline.close()
    OUT.write_text(json.dumps(records, indent=1))


asyncio.run(main())
