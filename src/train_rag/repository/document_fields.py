"""Queries for the mined fields of documents and their profile vectors."""

from collections.abc import Sequence

from sqlalchemy import and_, func
from sqlalchemy.dialects.postgresql import insert
from sqlmodel import Session, col, select

from train_rag.tables import DocumentFieldsRecord, DocumentRecord, FieldVectorRecord

FIELD_COLUMNS = ("doc_id", "source_type", "raw", "fields", "thread_kind", "status", "areas")


def document_keys(
    session: Session, dataset: str, doc_ids: Sequence[str]
) -> dict[str, list[tuple[int, str]]]:
    """Find the row ids and sources of documents by their dataset ids.

    Args:
        session: Active database session.
        dataset: Dataset of the documents.
        doc_ids: Dataset ids.

    Returns:
        (row id, source) of every row with each id; the dataset reuses a few ids.
    """
    statement = select(DocumentRecord.id, DocumentRecord.doc_id, DocumentRecord.source_type).where(
        DocumentRecord.dataset == dataset, col(DocumentRecord.doc_id).in_(doc_ids)
    )
    keys: dict[str, list[tuple[int, str]]] = {}
    for row_id, doc_id, source_type in session.exec(statement):
        if row_id is not None:
            keys.setdefault(doc_id, []).append((row_id, source_type))

    return keys


def upsert_document_fields(session: Session, records: Sequence[DocumentFieldsRecord]) -> None:
    """Store mined fields, replacing the same extractor's earlier fields of a document.

    Args:
        session: Active database session.
        records: Fields to store.
    """
    if not records:
        return

    rows = [record.model_dump(exclude={"id"}) for record in records]
    statement = insert(DocumentFieldsRecord)
    statement = statement.on_conflict_do_update(
        index_elements=["document_id", "extractor"],
        set_={name: statement.excluded[name] for name in (*FIELD_COLUMNS, "channel", "created_at")},
    )
    session.execute(statement, rows)


def fields_without_vectors(session: Session, model: str, limit: int) -> list[DocumentFieldsRecord]:
    """Load mined fields that have no profile vectors from a model yet.

    Args:
        session: Active database session.
        model: Embedding model of the vectors.
        limit: Rows to load.

    Returns:
        Up to `limit` rows, lowest id first.
    """
    statement = (
        select(DocumentFieldsRecord)
        .outerjoin(
            FieldVectorRecord,
            and_(
                col(FieldVectorRecord.document_fields_id) == col(DocumentFieldsRecord.id),
                col(FieldVectorRecord.model) == model,
            ),
        )
        .where(col(FieldVectorRecord.document_fields_id).is_(None))
        .order_by(col(DocumentFieldsRecord.id))
        .limit(limit)
    )

    return list(session.exec(statement))


def count_fields_without_vectors(session: Session, model: str) -> int:
    """Count mined fields that have no profile vectors from a model yet.

    Args:
        session: Active database session.
        model: Embedding model of the vectors.

    Returns:
        The number of rows.
    """
    with_vectors = select(FieldVectorRecord.document_fields_id).where(
        FieldVectorRecord.model == model
    )
    statement = (
        select(func.count())
        .select_from(DocumentFieldsRecord)
        .where(col(DocumentFieldsRecord.id).not_in(with_vectors))
    )

    return session.exec(statement).one()


def upsert_field_vectors(session: Session, records: Sequence[FieldVectorRecord]) -> None:
    """Store profile vectors, replacing a model's earlier vectors of the same fields.

    Args:
        session: Active database session.
        records: Vectors to store.
    """
    if not records:
        return

    rows = [record.model_dump() for record in records]
    statement = insert(FieldVectorRecord)
    statement = statement.on_conflict_do_update(
        index_elements=["document_fields_id", "model"],
        set_={name: statement.excluded[name] for name in ("subject", "problem", "names")},
    )
    session.execute(statement, rows)


def fields_with_vectors(
    session: Session, extractors: Sequence[str] | None, model: str
) -> list[tuple[DocumentFieldsRecord, FieldVectorRecord]]:
    """Load the mined fields of some extractors with their profile vectors.

    Args:
        session: Active database session.
        extractors: Extractors to load, or None for all; a document may have rows of
            several.
        model: Embedding model of the vectors; fields without them are left out.

    Returns:
        (fields, vectors) pairs, lowest fields id first.
    """
    statement = (
        select(DocumentFieldsRecord, FieldVectorRecord)
        .join(
            FieldVectorRecord,
            col(FieldVectorRecord.document_fields_id) == col(DocumentFieldsRecord.id),
        )
        .where(FieldVectorRecord.model == model)
        .order_by(col(DocumentFieldsRecord.id))
    )
    if extractors is not None:
        statement = statement.where(col(DocumentFieldsRecord.extractor).in_(extractors))

    return [(fields, vectors) for fields, vectors in session.exec(statement)]
