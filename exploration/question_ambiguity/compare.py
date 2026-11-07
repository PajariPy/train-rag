"""A web page to compare each sampled question's top 2 gte documents with its expected one.

Three columns side by side: gte #1, gte #2 and the expected document, with the question,
its gold answer and an ambiguity label on top. Each column starts with what that document
alone answers (from answers.py), and the passage it quotes is highlighted in the text.
Labels (ambiguous, not ambiguous or unsure) are saved to labels.json, the same file the
marimo labeller (label.py) uses.

Usage: uv run python exploration/question_ambiguity/compare.py [PORT]
"""

import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.processing.normalize import normalize_content
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.repository.question import load_questions

HERE = Path(__file__).parent
PAGE = HERE / "compare.html"
SAMPLE = HERE / "sample.json"
LABELS = HERE / "labels.json"
ANSWERS = HERE / "doc_answers.json"  # written by answers.py
SHUFFLE_SEED = 7  # the same order as label.py


class Label(BaseModel):
    """One question's ambiguity label."""

    label: str
    note: str = ""


def load_data() -> dict[str, object]:
    """The sampled questions, in shuffled order, and every document they show.

    Returns:
        `questions` (a list) and `documents` (keyed by dataset id).
    """
    sample = json.loads(SAMPLE.read_text())
    random.Random(SHUFFLE_SEED).shuffle(sample)
    wanted = {row["question_id"] for row in sample}
    with Session(get_engine()) as session:
        questions = {
            question.id: question
            for question in load_questions(session, EnterpriseRagBenchLoader.name)
            if question.id in wanted
        }
        doc_ids = {
            doc_id
            for row in sample
            for doc_id in [
                *questions[row["question_id"]].expected_doc_ids,
                *(hit["doc_id"] for hit in row["top2"]),
            ]
        }
        ids = ids_of_doc_ids(session, EnterpriseRagBenchLoader.name, sorted(doc_ids))
        documents = load_documents(session, sorted(ids))

    answers = json.loads(ANSWERS.read_text()) if ANSWERS.exists() else {}

    return {
        "answers": answers,
        "questions": [
            {
                "id": row["question_id"],
                "type": questions[row["question_id"]].question_type.value,
                "text": questions[row["question_id"]].text,
                "gold_answer": questions[row["question_id"]].gold_answer,
                "facts": questions[row["question_id"]].answer_facts,
                "expected": questions[row["question_id"]].expected_doc_ids,
                "top2": row["top2"],
            }
            for row in sample
        ],
        "documents": {
            document.doc_id: {
                "type": document.source_type,
                "name": document.title.splitlines()[0] if document.title else "",
                "chars": len(document.content),
                "text": normalize_content(document.content),
            }
            for document in documents
        },
    }


def read_labels() -> dict[str, dict[str, str]]:
    """The labels saved so far.

    Returns:
        Labels keyed by question id.
    """
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def create_app() -> FastAPI:
    """Build the app: the page at `/`, its data and labels under `/api`.

    Returns:
        The FastAPI app.
    """
    data = load_data()
    app = FastAPI(title="Compare documents")

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        """Serve the comparison page."""
        return PAGE.read_text()

    @app.get("/api/data")
    def get_data() -> dict[str, object]:
        """The questions and documents."""
        return data

    @app.get("/api/labels")
    def get_labels() -> dict[str, dict[str, str]]:
        """The labels saved so far."""
        return read_labels()

    @app.put("/api/labels/{question_id}")
    def put_label(question_id: str, label: Label) -> dict[str, dict[str, str]]:
        """Save one question's label and return all labels."""
        labels = read_labels()
        labels[question_id] = {
            "label": label.label,
            "note": label.note.strip(),
            "labelled_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        LABELS.write_text(json.dumps(labels, indent=1, sort_keys=True))

        return labels

    return app


def main() -> None:
    """Serve the page on this machine until stopped."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    print(f"Compare page: http://localhost:{port}")
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")


main()
