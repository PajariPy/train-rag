"""Queries for the documents table."""

from collections.abc import Sequence

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlmodel import Session, col, select

from train_rag.models import Artifact, SourceType
from train_rag.tables import DocumentRecord


def insert_documents(session: Session, documents: Sequence[DocumentRecord]) -> None:
    """Insert documents, skipping rows already imported from the same file and row.

    Args:
        session: Active database session.
        documents: Documents to insert.
    """
    if not documents:
        return

    rows = [document.model_dump(exclude={"id"}) for document in documents]
    statement = insert(DocumentRecord).on_conflict_do_nothing(
        index_elements=["dataset", "source_file", "row_index"]
    )
    session.execute(statement, rows)


def count_documents_by_source(session: Session, dataset: str) -> dict[str, int]:
    """Count stored documents per source type.

    Args:
        session: Active database session.
        dataset: Dataset to count documents of.

    Returns:
        Document counts keyed by source type.
    """
    statement = (
        select(DocumentRecord.source_type, func.count())
        .where(DocumentRecord.dataset == dataset)
        .group_by(DocumentRecord.source_type)
    )

    return {source_type: count for source_type, count in session.exec(statement)}


def get_document(session: Session, document_id: int) -> DocumentRecord | None:
    """Get the raw document a parsed document links to.

    Args:
        session: Active database session.
        document_id: Row id of the document.

    Returns:
        The document, or None when there is no such row.
    """
    return session.get(DocumentRecord, document_id)


def load_artifacts(
    session: Session, dataset: str, source_type: SourceType, limit: int | None = None
) -> list[Artifact]:
    """Load stored documents of one source as artifacts linked to their rows.

    Args:
        session: Active database session.
        dataset: Dataset to load documents of.
        source_type: Source to load.
        limit: Maximum number of documents, or None for all.

    Returns:
        Artifacts in row order, with `document_id` set.
    """
    statement = (
        select(DocumentRecord)
        .where(DocumentRecord.dataset == dataset, DocumentRecord.source_type == source_type)
        .order_by(col(DocumentRecord.id))
        .limit(limit)
    )

    return [to_artifact(document) for document in session.exec(statement)]


def sample_artifacts(
    session: Session, dataset: str, source_type: SourceType, size: int
) -> list[Artifact]:
    """Load a random sample of stored documents of one source.

    Args:
        session: Active database session.
        dataset: Dataset to sample from.
        source_type: Source to sample.
        size: Number of documents to sample.

    Returns:
        Up to `size` artifacts, with `document_id` set.
    """
    statement = (
        select(DocumentRecord)
        .where(DocumentRecord.dataset == dataset, DocumentRecord.source_type == source_type)
        .order_by(func.random())
        .limit(size)
    )

    return [to_artifact(document) for document in session.exec(statement)]


def to_artifact(document: DocumentRecord) -> Artifact:
    """Convert a documents row into an artifact that keeps the row id.

    Args:
        document: Stored document.

    Returns:
        The artifact, ready to be parsed.
    """
    return Artifact(
        id=document.doc_id,
        source_type=SourceType(document.source_type),
        title=document.title,
        content=document.content,
        source_file=document.source_file,
        row_index=document.row_index,
        document_id=document.id,
    )
