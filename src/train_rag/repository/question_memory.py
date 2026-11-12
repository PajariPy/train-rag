"""Queries for the question_memories table."""

from collections.abc import Sequence

from sqlalchemy.dialects.postgresql import insert
from sqlmodel import Session, col, select

from train_rag.models.memory import MemoryNote
from train_rag.tables import QuestionMemoryRecord


def upsert_memories(session: Session, records: Sequence[QuestionMemoryRecord]) -> None:
    """Store memory notes, replacing the stored note of a question that already has one.

    Args:
        session: Active database session.
        records: Notes to store.
    """
    if not records:
        return

    rows = [record.model_dump() for record in records]
    statement = insert(QuestionMemoryRecord)
    statement = statement.on_conflict_do_update(
        index_elements=["question_id"],
        set_={
            name: statement.excluded[name]
            for name in ("note", "wrong_detail", "model", "prompt_version", "created_at")
        },
    )
    session.execute(statement, rows)


def load_memories(session: Session, question_ids: Sequence[str]) -> dict[str, MemoryNote]:
    """Load the memory notes of questions.

    Args:
        session: Active database session.
        question_ids: Questions to load; ones without a note are left out.

    Returns:
        Notes keyed by question id.
    """
    statement = select(QuestionMemoryRecord).where(
        col(QuestionMemoryRecord.question_id).in_(question_ids)
    )

    return {
        record.question_id: MemoryNote(
            question_id=record.question_id, note=record.note, wrong_detail=record.wrong_detail
        )
        for record in session.exec(statement)
    }
