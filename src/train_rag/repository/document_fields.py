"""Queries for the mined fields of documents and their profile vectors."""

import typing
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from sqlalchemy import Text, and_, cast, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import QueryableAttribute, defer
from sqlmodel import Session, col, select

from train_rag.tables import DocumentFieldsRecord, DocumentRecord, FieldVectorRecord
from train_rag.tables.field_vector import PROFILE_DIMENSIONS

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

    Commits.

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
    session.commit()


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

    Commits.

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
    session.commit()


def load_document_fields(session: Session) -> list[DocumentFieldsRecord]:
    """Load every mined fields row, without the raw extractor output.

    Args:
        session: Active database session.

    Returns:
        The rows, lowest id first; `raw` is not loaded.
    """
    statement = (
        select(DocumentFieldsRecord)
        .options(defer(typing.cast(QueryableAttribute[Any], DocumentFieldsRecord.raw)))
        .order_by(col(DocumentFieldsRecord.id))
    )

    return list(session.exec(statement))


def profile_matrices(session: Session, model: str) -> dict[int, npt.NDArray[np.float32]]:
    """Load a model's profile vectors as arrays, fast.

    Vectors are read as text and parsed by numpy: building Python lists of floats for
    every vector takes minutes and gigabytes.

    Args:
        session: Active database session.
        model: Embedding model of the vectors.

    Returns:
        A (3, dimensions) float32 array (subject, problem, names; zeros for an empty
        profile) keyed by mined fields row id.
    """
    statement = select(
        FieldVectorRecord.document_fields_id,
        cast(col(FieldVectorRecord.subject), Text),
        cast(col(FieldVectorRecord.problem), Text),
        cast(col(FieldVectorRecord.names), Text),
    ).where(FieldVectorRecord.model == model)
    zero = np.zeros(PROFILE_DIMENSIONS, dtype=np.float32)

    def parse(value: str | None) -> npt.NDArray[np.float32]:
        """Parse one vector's text, "[0.1,0.2,...]".

        Args:
            value: The vector as Postgres writes it, or None for an empty profile.

        Returns:
            The vector.
        """
        if value is None:
            return zero

        return np.array(value[1:-1].split(","), dtype=np.float32)

    return {
        fields_id: np.stack([parse(subject), parse(problem), parse(names)])
        for fields_id, subject, problem, names in session.exec(statement)
    }
