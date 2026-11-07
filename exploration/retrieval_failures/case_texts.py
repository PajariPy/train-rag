"""Collect, for each missed question, what the case notes in post 003 show.

For every question: its text, the gold answer, the answers of the plain pipeline (run 8)
and the agent (run 14), the agent's extra searches with the documents each returned, and,
for each expected document, where it is in the dataset and the passage that best overlaps
the gold answer.

Usage: uv run python exploration/retrieval_failures/case_texts.py OUT.json QUESTION_ID...
"""

import json
import re
import sys
from pathlib import Path

from sqlalchemy import text
from sqlmodel import Session

from train_rag.database import get_engine

PIPELINE_RUN = 8
AGENT_RUN = 14
WINDOW = 700
WORD = re.compile(r"[a-z0-9][a-z0-9_.\-]*")


def words(value: str) -> set[str]:
    """Lowercase words longer than three characters."""
    return {word for word in WORD.findall(value.lower()) if len(word) > 3}


def best_passage(content: str, answer: str) -> str:
    """The window of the document sharing the most words with the gold answer."""
    wanted = words(answer)
    best, best_overlap = content[:WINDOW], -1
    for start in range(0, max(1, len(content) - WINDOW // 2), WINDOW // 4):
        window = content[start : start + WINDOW]
        overlap = len(wanted & words(window))
        if overlap > best_overlap:
            best, best_overlap = window, overlap
    return best.strip()


def answer(session: Session, run: int, question_id: str) -> str | None:
    """A run's answer to a question."""
    return session.execute(
        text("select answer from eval_results where run_id = :run and question_id = :q"),
        {"run": run, "q": question_id},
    ).scalar()


def title(session: Session, doc_id: str) -> str:
    """A document's source and the first line of its title."""
    source, value = session.execute(
        text("select source_type, title from documents where doc_id = :d"), {"d": doc_id}
    ).one()
    return f"{source}: {value.split(chr(10))[0][:100]}"


def agent_searches(session: Session, question_id: str) -> list[dict[str, object]]:
    """The searches the agent chose itself (run 14), with the documents each returned."""
    trace = session.execute(
        text("select trace from eval_results where run_id = :run and question_id = :q"),
        {"run": AGENT_RUN, "q": question_id},
    ).scalar()
    return [
        {
            "tool": step["tool"],
            "query": next(iter(step["arguments"].values()), ""),
            "refused": not step["doc_ids"],
            "documents": [
                {"doc_id": doc_id, "title": title(session, doc_id)} for doc_id in step["doc_ids"]
            ],
        }
        for step in trace or []
        if step["kind"] == "tool" and not step["arguments"].get("automatic")
    ]


def main() -> None:
    """Collect every question's case texts and write them to OUT."""
    out = Path(sys.argv[1])
    records = []
    with Session(get_engine()) as session:
        for question_id in sys.argv[2:]:
            question, gold, expected = session.execute(
                text("select text, gold_answer, expected_doc_ids from questions where id = :q"),
                {"q": question_id},
            ).one()
            documents = []
            for doc_id in expected:
                row = session.execute(
                    text(
                        "select title, content, char_count, source_file, row_index "
                        "from documents where doc_id = :d"
                    ),
                    {"d": doc_id},
                ).first()
                if row is None:
                    continue
                title, content, chars, source_file, row_index = row
                documents.append(
                    {
                        "doc_id": doc_id,
                        "title": title.split("\n")[0],
                        "chars": chars,
                        "source_file": source_file,
                        "row_index": row_index,
                        "passage": best_passage(content, gold),
                    }
                )
            records.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "gold_answer": gold,
                    "pipeline_answer": answer(session, PIPELINE_RUN, question_id),
                    "agent_answer": answer(session, AGENT_RUN, question_id),
                    "agent_searches": agent_searches(session, question_id),
                    "documents": documents,
                }
            )
    out.write_text(json.dumps(records, indent=1))


main()
