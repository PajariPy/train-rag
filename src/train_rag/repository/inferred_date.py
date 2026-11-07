"""Queries for meeting dates estimated by an LLM judge."""

from collections.abc import Sequence

from sqlmodel import Session, col, delete, select

from train_rag.models.parsed import InferredDate
from train_rag.tables import InferredDateRecord


def replace_inferred_dates(
    session: Session, method: str, dates: Sequence[tuple[int, InferredDate]]
) -> None:
    """Store a judge's dates, replacing the ones it stored before.

    Args:
        session: Active database session.
        method: Name of the judge that estimated the dates.
        dates: Pairs of document id and the date estimated for it.
    """
    session.exec(delete(InferredDateRecord).where(col(InferredDateRecord.method) == method))
    session.add_all(
        InferredDateRecord(document_id=document_id, method=method, **inferred.model_dump())
        for document_id, inferred in dates
    )
    session.commit()


def load_inferred_dates(session: Session) -> dict[int, InferredDate]:
    """Load every estimated meeting date, keyed by document id.

    Args:
        session: Active database session.

    Returns:
        The estimated date of each document that has one.
    """
    records = session.exec(select(InferredDateRecord))

    return {
        record.document_id: InferredDate.model_validate(record, from_attributes=True)
        for record in records
    }
