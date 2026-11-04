"""Queries for Slack messages split by an LLM."""

from collections.abc import Sequence

from sqlmodel import Session, col, delete, select

from train_rag.models.parsed import SlackMessage
from train_rag.tables import InferredMessagesRecord


def replace_inferred_messages(
    session: Session, method: str, threads: Sequence[tuple[int, list[SlackMessage]]]
) -> None:
    """Store a tool's message splits, replacing the ones it stored before.

    Args:
        session: Active database session.
        method: Name of the tool that split the threads.
        threads: Pairs of document id and the messages split from it.
    """
    session.exec(delete(InferredMessagesRecord).where(col(InferredMessagesRecord.method) == method))
    session.add_all(
        InferredMessagesRecord(
            document_id=document_id,
            messages=[message.model_dump() for message in messages],
            method=method,
        )
        for document_id, messages in threads
    )
    session.commit()


def load_inferred_messages(session: Session) -> dict[int, list[SlackMessage]]:
    """Load every inferred message split, keyed by document id.

    Args:
        session: Active database session.

    Returns:
        The messages of each thread that has a split.
    """
    return {
        record.document_id: [SlackMessage.model_validate(message) for message in record.messages]
        for record in session.exec(select(InferredMessagesRecord))
    }
