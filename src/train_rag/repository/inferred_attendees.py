"""Queries for meeting attendees listed by an LLM."""

from collections.abc import Sequence

from sqlmodel import Session, col, delete, select

from train_rag.models.parsed import Person
from train_rag.tables import InferredAttendeesRecord


def replace_inferred_attendees(
    session: Session, method: str, attendees: Sequence[tuple[int, list[Person]]]
) -> None:
    """Store a tool's attendee lists, replacing the ones it stored before.

    Args:
        session: Active database session.
        method: Name of the tool that listed the attendees.
        attendees: Pairs of document id and the people listed for it.
    """
    session.exec(
        delete(InferredAttendeesRecord).where(col(InferredAttendeesRecord.method) == method)
    )
    session.add_all(
        InferredAttendeesRecord(
            document_id=document_id,
            attendees=[person.model_dump() for person in people],
            method=method,
        )
        for document_id, people in attendees
    )
    session.commit()


def load_inferred_attendees(session: Session) -> dict[int, list[Person]]:
    """Load every inferred attendee list, keyed by document id.

    Args:
        session: Active database session.

    Returns:
        The people listed for each document that has a list.
    """
    return {
        record.document_id: [Person.model_validate(person) for person in record.attendees]
        for record in session.exec(select(InferredAttendeesRecord))
    }
