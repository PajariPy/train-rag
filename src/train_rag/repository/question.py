"""Queries for the questions table."""

from collections.abc import Sequence

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlmodel import Session, select

from train_rag.tables import QuestionRecord


def insert_questions(session: Session, questions: Sequence[QuestionRecord]) -> None:
    """Insert questions, skipping ids that are already stored.

    Args:
        session: Active database session.
        questions: Questions to insert.
    """
    if not questions:
        return

    rows = [question.model_dump() for question in questions]
    statement = insert(QuestionRecord).on_conflict_do_nothing(index_elements=["id"])
    session.execute(statement, rows)


def count_questions(session: Session, dataset: str) -> int:
    """Count stored questions.

    Args:
        session: Active database session.
        dataset: Dataset to count questions of.

    Returns:
        The number of questions.
    """
    statement = (
        select(func.count()).select_from(QuestionRecord).where(QuestionRecord.dataset == dataset)
    )

    return session.exec(statement).one()
