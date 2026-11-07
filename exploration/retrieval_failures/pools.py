"""Write each problem case's candidate pool, with every document's text, for outside models.

The pool is the one the smaller-chunk test used (gte_small_chunks.py): gte's own top 50 on
the stored chunks, plus the expected document and the pipeline's picks. Writing it to one
file lets a script run models that need a different environment (jina_late_chunking.py
needs transformers 4) without the project's database code.

Usage: uv run python exploration/retrieval_failures/pools.py OUT.json FAILURES.json QUESTION_ID...
"""

import asyncio
import json
import sys
from pathlib import Path

from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.pipelines.retrieval import RetrievalPipeline
from train_rag.processing.normalize import normalize_documents, normalize_text
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.repository.question import load_questions

OUT = Path(sys.argv[1])
FAILURES = Path(sys.argv[2])
QUESTION_IDS = sys.argv[3:]


async def gte_top50(questions: dict[str, str]) -> dict[str, list[str]]:
    """The documents of gte's own top 50 for every question, best first."""
    pipeline = RetrievalPipeline.load(rerank=False)
    pools = {}
    try:
        for qid, text in questions.items():
            hits = await pipeline.semantic_search(normalize_text(text))
            pools[qid] = [hit.doc_id for hit in hits]
    finally:
        await pipeline.close()

    return pools


def main() -> None:
    """Build every pool and write the pools with their documents' text."""
    records = {r["question_id"]: r for r in json.loads(FAILURES.read_text())}
    engine = get_engine()
    with Session(engine) as session:
        loaded = load_questions(session, EnterpriseRagBenchLoader.name, None, None)
        texts = {q.id: q.text for q in loaded if q.id in QUESTION_IDS}
    top50 = asyncio.run(gte_top50(texts))

    pools = {}
    for qid in QUESTION_IDS:
        record = records[qid]
        expected = record["expected"][0]["doc_id"]
        picks = [pick["doc_id"] for pick in record["picked"]]
        pools[qid] = {
            "question": texts[qid],
            "expected": expected,
            "gte_top50": top50[qid],
            "members": list(dict.fromkeys([expected, *top50[qid], *picks])),
        }
    unique = sorted({doc_id for pool in pools.values() for doc_id in pool["members"]})
    with Session(engine) as session:
        internal = ids_of_doc_ids(session, EnterpriseRagBenchLoader.name, unique)
        clean = normalize_documents(load_documents(session, sorted(internal)))
    documents = {doc.doc_id: f"{doc.title}\n\n{doc.text}" for doc in clean}
    OUT.write_text(json.dumps({"pools": pools, "documents": documents}))
    print(f"wrote {len(pools)} pools, {len(documents)} documents to {OUT}", flush=True)


main()
