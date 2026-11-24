"""A web page to review the field agent's semantic questions, one at a time, with notes.

For every question of the logged run (`evaluate-search --retriever field-agent --log`): where
the expected document ended (agent rank, first-search rank), every search the agent made with
its query, settings and top results, what it picked, the check's verdict and reasoning, the
final documents, and the whole conversation. Notes are saved to `review_notes.json`.

Reads `data/run-2026-11-24.jsonl` (the agent's log) and `data/baseline.json` (baseline.py;
optional). Nothing is paid for.

Usage: uv run python exploration/field_agent_review/review.py [PORT]
"""

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
from train_rag.models import Question, QuestionType
from train_rag.processing.normalize import normalize_content
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.repository.question import load_questions

HERE = Path(__file__).parent
DATA = HERE / "data"
LOG = DATA / "run-2026-11-24.jsonl"
BASELINE = DATA / "baseline.json"
PAGE = HERE / "review.html"
NOTES = HERE / "review_notes.json"  # your review notes, kept out of the ignored data/
SHOWN = 10  # results the agent saw per search
OUTCOMES = {
    "first": "found first",
    "top5": "found at 2–5",
    "top20": "found at 6–20",
    "missed": "not in the top 20",
}

Record = dict[str, Any]


def rank_of(expected: list[str], ranking: list[str]) -> int | None:
    """Find the best rank (1-based) of any expected document in a ranking.

    Args:
        expected: The expected doc ids.
        ranking: Doc ids, best first.

    Returns:
        The rank, or None when none is there.
    """
    ranks = [ranking.index(doc_id) + 1 for doc_id in expected if doc_id in ranking]

    return min(ranks) if ranks else None


def outcome(rank: int | None) -> str:
    """Name the group of an agent rank.

    Args:
        rank: The expected document's rank, or None.

    Returns:
        A key of OUTCOMES.
    """
    if rank is None:
        return "missed"
    if rank == 1:
        return "first"

    return "top5" if rank <= 5 else "top20"


def thoughts_before_calls(trace: list[Record]) -> list[str]:
    """Pair each tool call with the visible text and thoughts of the turn that made it.

    Args:
        trace: The run's trace steps.

    Returns:
        One entry per tool call in the trace, in order.
    """
    paired = []
    for step in trace:
        if step["kind"] == "model":
            said = "\n".join(part for part in (step["thoughts"], step["text"]) if part.strip())
            paired += [said] * len(step["tool_calls"])

    return paired


def searches_of(record: Record, expected: list[str]) -> list[Record]:
    """Describe each search: tool, query, settings, the agent's reasoning, its results.

    Args:
        record: One logged question.
        expected: The expected doc ids.

    Returns:
        One dict per search, in order.
    """
    trace = record["run"]["trace"]
    calls = [step for step in trace if step["kind"] == "model"]
    reasons = thoughts_before_calls(trace)
    tool_names = [call["tool"] for step in calls for call in step["tool_calls"]]
    search_reasons = [
        reason
        for reason, tool in zip(reasons, tool_names, strict=False)
        if tool in ("search_documents", "search_fields")
    ]
    searches = []
    for place, (settings, ranking) in enumerate(
        zip(record["run"]["searches"], record["searches"].values(), strict=False)
    ):
        searches.append(
            {
                **settings,
                "reason": search_reasons[place] if place < len(search_reasons) else "",
                "results": ranking[:SHOWN],
                "expected_rank": rank_of(expected, ranking),
                "logged": len(ranking),
            }
        )

    return searches


def build(
    records: list[Record], questions: dict[str, Question], baseline: dict[str, list[str]]
) -> tuple[list[Record], dict[str, Record], set[str]]:
    """Summarise every logged question and gather the doc ids the page shows.

    Args:
        records: The agent's log, one record per question.
        questions: The semantic questions by text.
        baseline: The first search's top 50 per question id.

    Returns:
        The list summaries, the details by question id, and the doc ids to load.
    """
    summaries, details, doc_ids = [], {}, set()
    for record in records:
        question = questions[record["question"]]
        expected = question.expected_doc_ids
        agent_rank = rank_of(expected, record["ranking"])
        first_rank = rank_of(expected, baseline.get(question.id, []))
        searches = searches_of(record, expected)
        scores = record["scores"]
        seen_by = [place + 1 for place, s in enumerate(searches) if s["expected_rank"]]
        kind = outcome(agent_rank)
        summary = {
            "id": question.id,
            "outcome": kind,
            "outcome_label": OUTCOMES[kind],
            "source": question.source_types[0].value,
            "agent_rank": agent_rank,
            "first_rank": first_rank if question.id in baseline else "not run",
            "searches": len(searches),
            "seen": bool(seen_by),
        }
        top_score = max(scores.values()) if scores else None
        best_expected = max((scores[d] for d in expected if d in scores), default=None)
        summaries.append(summary)
        details[question.id] = {
            **summary,
            "question": question.text,
            "gold_answer": question.gold_answer,
            "expected": expected,
            "seen_by": seen_by,
            "picked": [d for d in record["run"]["picks"] if d in expected],
            "picks": record["run"]["picks"],
            "candidates": record["run"]["candidates"],
            "expected_score": best_expected,
            "top_score": top_score,
            "reranked": len(scores),
            "searches_detail": searches,
            "checks": record["run"]["checks"],
            "final": record["final"],
            "ranking": record["ranking"],
            "baseline": baseline.get(question.id, [])[:20],
            "trace": record["run"]["trace"],
        }
        doc_ids |= {
            *expected,
            *record["ranking"],
            *record["run"]["picks"],
            *(d for s in searches for d in s["results"]),
            *baseline.get(question.id, [])[:20],
        }

    return summaries, details, doc_ids


def doc_name(document: Any) -> str:  # noqa: ANN401 - a DocumentRecord or None
    """Name a document by its source and the first line of its title.

    Args:
        document: The document, or None.

    Returns:
        E.g. "[slack] #billing thread".
    """
    if document is None:
        return "unknown document"

    return f"[{document.source_type}] {document.title.splitlines()[0]}"


def load() -> tuple[list[Record], dict[str, Record], dict[str, str], dict[str, str]]:
    """Read the log, the questions and the documents the page shows.

    Returns:
        Summaries, details, document names and document texts.
    """
    records = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    with Session(get_engine()) as session:
        dataset = EnterpriseRagBenchLoader.name
        questions = {
            q.text: q for q in load_questions(session, dataset, QuestionType.SEMANTIC, None)
        }
        summaries, details, doc_ids = build(records, questions, baseline)
        ids = ids_of_doc_ids(session, dataset, sorted(doc_ids))
        documents = {d.doc_id: d for d in load_documents(session, sorted(ids))}

    names = {doc_id: doc_name(documents.get(doc_id)) for doc_id in doc_ids}
    texts = {doc_id: normalize_content(doc.content) for doc_id, doc in documents.items()}

    return summaries, details, names, texts


class Note(BaseModel):
    """A review note on one question."""

    text: str


def load_notes() -> dict[str, dict[str, str]]:
    """Read every saved review note.

    Returns:
        Notes keyed by question id.
    """
    return json.loads(NOTES.read_text()) if NOTES.exists() else {}


SUMMARIES, DETAILS, NAMES, TEXTS = load()
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Serve the page."""
    return PAGE.read_text()


@app.get("/api/cases")
def list_cases() -> list[Record]:
    """List every question, for the side list."""
    return SUMMARIES


@app.get("/api/case/{qid}")
def case(qid: str) -> Record:
    """Return everything about one question, with document names."""
    if qid not in DETAILS:
        raise HTTPException(404, f"no question {qid}")
    detail = DETAILS[qid]
    shown = {
        *detail["expected"],
        *detail["ranking"],
        *detail["picks"],
        *detail["baseline"],
        *(d for s in detail["searches_detail"] for d in s["results"]),
    }

    return {**detail, "names": {doc_id: NAMES.get(doc_id, doc_id) for doc_id in shown}}


@app.get("/api/doc/{doc_id}")
def document(doc_id: str) -> dict[str, str]:
    """Return one document's full normalized text."""
    if doc_id not in TEXTS:
        raise HTTPException(404, f"no document {doc_id}")

    return {"text": TEXTS[doc_id]}


@app.get("/api/notes")
def notes() -> dict[str, dict[str, str]]:
    """Return every saved note, for the page to show and mark in the list."""
    return load_notes()


@app.put("/api/note/{qid}")
def save_note(qid: str, note: Note) -> dict[str, str]:
    """Save one question's note; an empty note removes it.

    The file is rewritten through a temporary file, so a crash never leaves it half written.
    """
    if qid not in DETAILS:
        raise HTTPException(404, f"no question {qid}")
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


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8768
    print(f"Field agent review: http://localhost:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
