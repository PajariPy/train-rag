"""Queries for the questions table."""

from collections.abc import Sequence

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlmodel import Session, select

from train_rag.models import Question, QuestionType, SourceType
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


def load_questions(
    session: Session,
    dataset: str,
    question_type: QuestionType | None = None,
    limit: int | None = None,
) -> list[Question]:
    """Load benchmark questions, optionally of one type.

    Args:
        session: Active database session.
        dataset: Dataset to load questions of.
        question_type: Only questions of this type, or None for all.
        limit: Maximum number of questions, or None for all.

    Returns:
        The questions ordered by id.
    """
    statement = select(QuestionRecord).where(QuestionRecord.dataset == dataset)
    if question_type is not None:
        statement = statement.where(QuestionRecord.question_type == question_type)
    statement = statement.order_by(QuestionRecord.id).limit(limit)

    return [to_question(record) for record in session.exec(statement)]


def to_question(record: QuestionRecord) -> Question:
    """Convert a questions row into a question.

    Args:
        record: Stored question.

    Returns:
        The question.
    """
    return Question(
        id=record.id,
        question_type=QuestionType(record.question_type),
        text=record.text,
        gold_answer=record.gold_answer,
        answer_facts=record.answer_facts,
        source_types=[SourceType(source) for source in record.source_types],
        expected_doc_ids=record.expected_doc_ids,
    )
