"""Queries for the parsed documents table."""

from collections.abc import Sequence

from sqlalchemy.dialects.postgresql import insert
from sqlmodel import Session

from train_rag.models.parsed import ParsedDocument
from train_rag.tables import ParsedDocumentRecord

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


def without_nul(value: JsonValue) -> JsonValue:
    """Remove NUL characters, which Postgres text and JSONB cannot store.

    Args:
        value: JSON-like value: a string, list, dict or scalar.

    Returns:
        The same value with every NUL removed from its strings.
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [without_nul(item) for item in value]
    if isinstance(value, dict):
        return {key: without_nul(item) for key, item in value.items()}

    return value


def upsert_parsed_documents(session: Session, documents: Sequence[ParsedDocument]) -> None:
    """Store parsed documents, replacing an earlier parse of the same raw document.

    Does not commit, so the parse can be committed together with the chunks.

    Args:
        session: Active database session.
        documents: Parsed documents with `document_id` set.
    """
    rows = [
        {
            "document_id": document.document_id,
            "doc_id": document.doc_id,
            "source_type": document.source_type.value,
            "kind": type(document).__name__,
            "useful": document.useful,
            "flagged": document.flagged,
            "data": without_nul(document.model_dump(mode="json")),
        }
        for document in documents
        if document.document_id is not None
    ]
    if not rows:
        return

    statement = insert(ParsedDocumentRecord)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=["document_id"],
            set_={
                column: statement.excluded[column]
                for column in ("doc_id", "source_type", "kind", "useful", "flagged", "data")
            },
        ),
        rows,
    )
