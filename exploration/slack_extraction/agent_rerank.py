"""Rerank the agent's candidates: how many right threads end up in the final top 5?

Agent v3 (agent_search.py, data/agent_search_documented_result.json) gave each of the 200
Slack dev questions a ranked list: its picks first, then its searches fused. The lists are
rebuilt from the saved searches (deterministic, no DeepSeek). The reranker then orders:

- today:            today's candidates (three search lists pooled), as in the baseline;
- agent 50 / 100:   the agent's top 50 / 100 only;
- today + agent N:  both lists together.

Documents already in today's list keep their baseline reranker score. A new document is
shown to the reranker like the pipeline does: title + its stored chunk closest to the
question (fine-tuned gte). Same caveat as before: the agent searched the 24k labelled pool.

Free and local (GPU). Usage:
    uv run python exploration/slack_extraction/agent_rerank.py
"""

import json
from typing import Any

import numpy as np
from agent_search import QUESTIONS, RRF_K, FieldIndex
from field_match import BASELINE, HERE
from sqlalchemy import text
from sqlmodel import Session, select

from train_rag.database import get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.models import SourceType
from train_rag.pipelines.data_ingestion import FIELD
from train_rag.processing.normalize import normalize_text
from train_rag.reranker import GteModernBertReranker
from train_rag.tables import DocumentRecord
from train_rag.tables.embedding import FINETUNED_TABLES

AGENT = HERE / "data" / "agent_search_documented_result.json"
SCORES = HERE / "data" / "agent_rerank_scores.json"
RESULT = HERE / "data" / "agent_rerank_result.json"
DEPTHS = (50, 100)


def agent_lists(index: FieldIndex, runs: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Rebuild each question's agent list: picks first, then the searches fused by rank."""
    lists: dict[str, list[str]] = {}

    for run in runs:
        fused: dict[int, float] = {}

        for search in run["searches"]:
            found = index.search(search["query"], search["profile"], search["areas"])

            for rank, position in enumerate(found):
                fused[position] = fused.get(position, 0.0) + 1 / (RRF_K + rank + 1)

        ranked = [index.pool[p] for p in sorted(fused, key=lambda p: -fused[p])]
        lists[run["doc_id"]] = list(dict.fromkeys(run["picked"] + ranked))

    return lists


def best_chunks(needed: dict[str, set[str]], questions: dict[str, str]) -> dict[tuple[str, str], str]:
    """The chunk text closest to the question, per (question, new document)."""
    docs = sorted({doc for found in needed.values() for doc in found})
    order = sorted(needed)
    vectors = np.asarray(FineTunedGteEmbedder(batch_size=64).embed([questions[q] for q in order]))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    q_vec = dict(zip(order, vectors, strict=True))
    chunks: dict[str, list[tuple[np.ndarray, str]]] = {}

    with get_engine().connect() as connection:
        for source in SourceType:
            table = FINETUNED_TABLES[source].__tablename__
            rows = connection.execute(
                text(
                    f"SELECT doc_id, text, embedding::text FROM {table} "
                    "WHERE doc_id = ANY(:ids) AND field = :field"
                ),
                {"ids": docs, "field": FIELD},
            ).all()

            for doc_id, chunk, embedding in rows:
                vector = np.array(json.loads(embedding), dtype=np.float32)
                chunks.setdefault(doc_id, []).append((vector / np.linalg.norm(vector), chunk))

    best: dict[tuple[str, str], str] = {}
    for question, found in needed.items():
        for doc in found:
            if doc in chunks:
                best[(question, doc)] = max(chunks[doc], key=lambda c: float(c[0] @ q_vec[question]))[1]

    return best


def hits(ranks: list[int | None]) -> dict[str, float]:
    """Top 1 / 5 / 10 in % of the questions."""
    return {
        f"top{k}": round(100 * sum(1 for r in ranks if r is not None and r <= k) / len(ranks), 1)
        for k in (1, 5, 10)
    }


def main() -> None:
    """Score the new candidates with the reranker and compare the candidate sets."""
    baseline: dict[str, dict[str, Any]] = json.loads(BASELINE.read_text())
    runs = [r for r in json.loads(AGENT.read_text())["runs"] if "error" not in r]
    rows = [json.loads(line) for line in QUESTIONS.open()]
    questions = {row["doc_id"]: row["question"] for row in rows if row["split"] == "dev_slack"}
    lists = agent_lists(FieldIndex(), runs)

    scores: dict[str, dict[str, float]] = {q: dict(baseline[q]["scores"]) for q in lists}
    cached = json.loads(SCORES.read_text()) if SCORES.exists() else {}
    needed = {
        q: {d for d in found[: max(DEPTHS)] if d not in scores[q] and d not in cached.get(q, {})}
        for q, found in lists.items()
    }
    total = sum(len(found) for found in needed.values())
    print(f"{total:,} new documents to rerank", flush=True)

    if total:
        chunk = best_chunks(needed, questions)
        with Session(get_engine()) as session:
            titles = dict(
                session.exec(
                    select(DocumentRecord.doc_id, DocumentRecord.title).where(
                        DocumentRecord.doc_id.in_(sorted({d for f in needed.values() for d in f}))
                    )
                ).all()
            )
        reranker = GteModernBertReranker(batch_size=8)

        for number, (q, found) in enumerate(needed.items(), 1):
            docs = [d for d in sorted(found) if (q, d) in chunk]
            texts = [f"{titles.get(d) or ''}\n{chunk[(q, d)]}" for d in docs]
            cached.setdefault(q, {}).update(
                zip(docs, reranker.score(normalize_text(questions[q]), texts), strict=True)
            )
            if number % 20 == 0:
                print(f"  reranked {number}/{len(needed)} questions", flush=True)
                SCORES.write_text(json.dumps(cached))

        SCORES.write_text(json.dumps(cached))

    for q in scores:
        scores[q] |= cached.get(q, {})

    def evaluate(pick: Any) -> dict[str, Any]:
        ranks, reached, size = [], 0, []
        for q in lists:
            candidates = pick(q)
            size.append(len(candidates))
            reached += q in candidates
            order = sorted(candidates, key=lambda d: -scores[q].get(d, -1e9))
            ranks.append(order.index(q) + 1 if q in order else None)
        return {"reaches reranker": round(100 * reached / len(lists), 1),
                "candidates": int(np.mean(size))} | hits(ranks)

    report = {"today": evaluate(lambda q: set(baseline[q]["scores"]))}
    for depth in DEPTHS:
        report[f"agent {depth}"] = evaluate(lambda q, n=depth: set(lists[q][:n]))
        report[f"today + agent {depth}"] = evaluate(
            lambda q, n=depth: set(baseline[q]["scores"]) | set(lists[q][:n])
        )

    RESULT.write_text(json.dumps(report, indent=1))
    for name, row in report.items():
        print(f"{name:20s} {row}")


if __name__ == "__main__":
    main()
