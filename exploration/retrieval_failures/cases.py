"""A web page to read the wrong answers of the cited run, one case at a time.

The same data as stage 2 of notebook 006, in a larger, quieter layout: for every wrong answer
of CITED_RUN, its outcome (from the per-document ratings of the second judge), where the
expected document ranks in BM25 and gte, the reranker's view, the smaller-chunk test, the
documents with their ratings and full text, the pipeline's current top 5, and the agent's
own searches.

Reads `data/cases_run15.json`, `data/failures_run15.json` and `data/gte_small_chunks.json`
(written by case_texts.py, analyze.py and gte_small_chunks.py) and the rating cache in
`exploration/question_ambiguity/`. Nothing is paid for: ratings that aren't cached are shown
as not rated.

Usage: uv run python exploration/retrieval_failures/cases.py [PORT]
"""

import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.models import QuestionType
from train_rag.processing.normalize import normalize_content
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.repository.evaluation import results_of_run
from train_rag.repository.question import load_questions

HERE = Path(__file__).parent
DATA = HERE / "data"
PAGE = HERE / "cases.html"
NOTES = HERE / "review_notes.json"  # your review notes, kept out of the ignored data/
CITED_RUN = 15
AGENT_RUN = 14
OUTCOMES = {
    "retrieved": "expected document retrieved",
    "full": "cited document answers fully",
    "partial": "cited document answers partly",
    "none": "cited document doesn't answer",
    "uncited": "no document cited",
}
# Local module, loaded by name from its folder (as notebook 006 does).
sys.path.insert(0, str(HERE.parent / "question_ambiguity"))
RatingCache = importlib.import_module("answers").RatingCache


def read_json(name: str) -> Any:  # noqa: ANN401 - free-form analysis files
    """One of the analysis files in DATA, or an empty dict when it isn't written yet."""
    path = DATA / name
    return json.loads(path.read_text()) if path.exists() else {}


def judge(result: Any, facts: list[str]) -> dict[str, Any] | None:  # noqa: ANN401
    """A result's judge verdict and which gold facts the answer covered."""
    if result is None:
        return None

    return {
        "correct": result.correct,
        "completeness": result.completeness,
        "score": result.score,
        "reasoning": result.judge_reasoning,
        "facts": [
            {"fact": fact, "covered": bool(hit)}
            for fact, hit in zip(facts, result.facts_covered, strict=False)
        ],
    }


def outcome(expected: list[str], retrieved: list[str], cited: list[str], ratings: dict) -> str:
    """Sort one wrong answer by the document it was taken from (stage 2 of notebook 006)."""
    if set(expected) & set(retrieved):
        return "retrieved"
    if not cited:
        return "uncited"
    coverage = {ratings[doc_id]["coverage"] for doc_id in cited if doc_id in ratings}
    for level in ("full", "partial"):
        if level in coverage:
            return level

    return "none"


def best_coverage(coverages: list[str]) -> str | None:
    """The best of some ratings (full, then partial, then none), or None if none are rated."""
    for level in ("full", "partial", "none"):
        if level in coverages:
            return level

    return None


def build() ->tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    """Every wrong answer's summary and details, and the text of every document they show."""
    cases = {row["question_id"]: row for row in read_json("cases_run15.json") or []}
    measured = {row["question_id"]: row for row in read_json("failures_run15.json") or []}
    chunks = {row["question_id"]: row for row in read_json("gte_small_chunks.json") or []}
    ratings_cache = RatingCache()
    with Session(get_engine()) as session:
        dataset = EnterpriseRagBenchLoader.name
        questions = {q.id: q for q in load_questions(session, dataset, QuestionType.SEMANTIC)}
        wrong = {
            r.question_id: r
            for r in results_of_run(session, CITED_RUN)
            if r.correct is False and r.question_id in questions
        }
        agent = {r.question_id: r for r in results_of_run(session, AGENT_RUN)}
        doc_ids = {
            doc_id
            for qid, result in wrong.items()
            for doc_id in [
                *result.retrieved_doc_ids,
                *questions[qid].expected_doc_ids,
                *(p["doc_id"] for p in measured.get(qid, {}).get("picked", [])),
                *(d["doc_id"] for s in cases.get(qid, {}).get("agent_searches", [])
                  for d in s["documents"]),
            ]
        }
        ids = ids_of_doc_ids(session, dataset, sorted(doc_ids))
        documents = {d.doc_id: d for d in load_documents(session, sorted(ids))}

    texts = {doc_id: normalize_content(doc.content) for doc_id, doc in documents.items()}
    summaries, details = [], {}
    for qid, result in sorted(wrong.items()):
        question = questions[qid]
        expected = question.expected_doc_ids
        shown = list(dict.fromkeys([*result.retrieved_doc_ids, *expected]))
        ratings = {
            doc_id: rating.model_dump()
            for doc_id in shown
            if doc_id in documents
            and (rating := ratings_cache.rating(question.text, documents[doc_id])) is not None
        }
        kind = outcome(expected, result.retrieved_doc_ids, result.cited_doc_ids, ratings)
        record = measured.get(qid)
        target = record["expected"][0] if record and record["expected"] else None
        summary = {
            "id": qid,
            "outcome": kind,
            "outcome_label": OUTCOMES[kind],
            "source": question.source_types[0].value,
            "bm25_rank": target["bm25_rank"] if target else None,
            "gte_rank": target["gte_rank"] if target else None,
            "given_best": best_coverage(
                [ratings[d]["coverage"] for d in result.retrieved_doc_ids if d in ratings]
            ),
            "expected_coverage": best_coverage(
                [ratings[d]["coverage"] for d in expected if d in ratings]
            ),
        }
        summaries.append(summary)
        details[qid] = {
            **summary,
            "question": question.text,
            "gold_answer": question.gold_answer,
            "answer": result.answer,
            "judge": judge(result, question.answer_facts),
            "documents": [
                {
                    "doc_id": doc_id,
                    "name": doc_name(documents.get(doc_id)),
                    "chars": len(texts.get(doc_id, "")),
                    "rank": result.retrieved_doc_ids.index(doc_id) + 1
                    if doc_id in result.retrieved_doc_ids
                    else None,
                    "cited": doc_id in result.cited_doc_ids,
                    "expected": doc_id in expected,
                    "rating": ratings.get(doc_id),
                }
                for doc_id in shown
            ],
            "measured": record,
            "expected_doc": (cases.get(qid, {}).get("documents") or [None])[0],
            "small_chunks": chunks.get(qid),
            "agent_searches": [
                {**search, "documents": [
                    {**d, "name": doc_name(documents.get(d["doc_id"])),
                     "expected": d["doc_id"] in expected}
                    for d in search["documents"]
                ]}
                for search in cases.get(qid, {}).get("agent_searches", [])
            ] if qid in agent else None,
            "agent_answer": agent[qid].answer if qid in agent else None,
            "agent_judge": judge(agent.get(qid), question.answer_facts),
        }

    return summaries, details, texts


def doc_name(document: Any) -> str:  # noqa: ANN401
    """A document's type and name (the first line of its title)."""
    if document is None:
        return "unknown document"

    return f"[{document.source_type}] {document.title.splitlines()[0]}"


class Note(BaseModel):
    """A review note on one case."""

    text: str


def load_notes() -> dict[str, dict[str, str]]:
    """Every saved review note, keyed by question id."""
    return json.loads(NOTES.read_text()) if NOTES.exists() else {}


SUMMARIES, DETAILS, TEXTS = build()
app = FastAPI()


@app.get("/api/notes")
def notes() -> dict[str, dict[str, str]]:
    """Every saved review note, for the page to show and mark in the list."""
    return load_notes()


@app.put("/api/note/{qid}")
def save_note(qid: str, note: Note) -> dict[str, str]:
    """Save one case's note; an empty note removes it.

    The file is rewritten through a temporary file, so a crash never leaves it half written.
    """
    if qid not in DETAILS:
        raise HTTPException(404, f"no wrong answer {qid}")
    saved = load_notes()
    text = note.text.strip()
    if text:
        saved[qid] = {"text": text, "updated": datetime.now(UTC).isoformat(timespec="seconds")}
    else:
        saved.pop(qid, None)
    partial = NOTES.with_suffix(".json.tmp")
    partial.write_text(json.dumps(dict(sorted(saved.items())), indent=1, ensure_ascii=False))
    partial.replace(NOTES)

    return saved.get(qid, {"text": "", "updated": ""})


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """The page."""
    return PAGE.read_text()


@app.get("/api/cases")
def list_cases() -> list[dict[str, Any]]:
    """Every wrong answer, for the side list."""
    return SUMMARIES


@app.get("/api/case/{qid}")
def case(qid: str) -> dict[str, Any]:
    """Everything about one wrong answer."""
    if qid not in DETAILS:
        raise HTTPException(404, f"no wrong answer {qid}")

    return DETAILS[qid]


@app.get("/api/doc/{doc_id}")
def document(doc_id: str) -> dict[str, str]:
    """One document's full normalized text."""
    if doc_id not in TEXTS:
        raise HTTPException(404, f"no document {doc_id}")

    return {"text": TEXTS[doc_id]}


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8767
    print(f"Cases page: http://localhost:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
