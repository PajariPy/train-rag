"""Queries for the embedding tables: store vectors, index them and search them."""

from collections.abc import Sequence

from sqlalchemy import insert, text
from sqlmodel import Session, col, delete, select

from train_rag.models import SourceType
from train_rag.models.chunk import Chunk
from train_rag.models.retrieval import SearchHit
from train_rag.repository.document import document_lengths
from train_rag.tables import EMBEDDING_TABLES, EmbeddingRecord

INSERT_BATCH = 2_000
INDEX_MEMORY = "2GB"


def delete_embeddings(
    session: Session, table: type[EmbeddingRecord], model: str, field: str
) -> None:
    """Delete one model's embeddings of one field from a table.

    Args:
        session: Active database session.
        table: Embedding table, from `EMBEDDING_TABLES`.
        model: Embedding model name.
        field: Embedded field, e.g. `summary`.
    """
    session.exec(delete(table).where(col(table.model) == model, col(table.field) == field))
    session.commit()


def delete_document_embeddings(
    session: Session,
    table: type[EmbeddingRecord],
    model: str,
    field: str,
    document_ids: Sequence[int],
) -> int:
    """Delete one model's embeddings of one field for some documents of a table.

    Args:
        session: Active database session.
        table: Embedding table, from `EMBEDDING_TABLES`.
        model: Embedding model name.
        field: Embedded field, e.g. `chunk`.
        document_ids: Row ids of the documents.

    Returns:
        The number of rows deleted.
    """
    deleted = 0
    for start in range(0, len(document_ids), INSERT_BATCH):
        result = session.exec(
            delete(table).where(
                col(table.model) == model,
                col(table.field) == field,
                col(table.document_id).in_(document_ids[start : start + INSERT_BATCH]),
            )
        )
        deleted += result.rowcount
    session.commit()

    return deleted


def insert_embeddings(
    session: Session, table: type[EmbeddingRecord], records: Sequence[EmbeddingRecord]
) -> None:
    """Insert embedding rows in batches, without committing.

    Rows keep an `id` set beforehand (see `reserve_ids`); otherwise the database picks it.

    Args:
        session: Active database session.
        table: Embedding table the rows belong to.
        records: Rows built with `table`.
    """
    for start in range(0, len(records), INSERT_BATCH):
        batch = records[start : start + INSERT_BATCH]
        session.execute(
            insert(table),
            [record.model_dump(exclude={"id"} if record.id is None else set()) for record in batch],
        )


def reserve_ids(session: Session, table: type[EmbeddingRecord], count: int) -> list[int]:
    """Take row ids from a table's id sequence before inserting the rows.

    Knowing the ids up front lets each chunk store its neighbours' ids in the same insert.

    Args:
        session: Active database session.
        table: Embedding table.
        count: Number of ids.

    Returns:
        New, unused row ids in increasing order.
    """
    statement = text(
        "SELECT nextval(pg_get_serial_sequence(:table, 'id')) FROM generate_series(1, :count)"
    )
    rows = session.connection().execute(statement, {"table": table.__tablename__, "count": count})

    return [int(row[0]) for row in rows]


def replace_embeddings(
    session: Session,
    table: type[EmbeddingRecord],
    model: str,
    field: str,
    records: Sequence[EmbeddingRecord],
) -> None:
    """Store one model's embeddings of one field, replacing the ones stored before.

    Args:
        session: Active database session.
        table: Embedding table, from `EMBEDDING_TABLES`.
        model: Embedding model name.
        field: Embedded field, e.g. `summary`.
        records: Rows built with `table`.
    """
    delete_embeddings(session, table, model, field)
    insert_embeddings(session, table, records)
    session.commit()


def embedded_document_ids(
    session: Session, table: type[EmbeddingRecord], model: str, field: str
) -> set[int]:
    """Find the documents that already have embeddings of a field, to skip them.

    Args:
        session: Active database session.
        table: Embedding table.
        model: Embedding model name.
        field: Embedded field.

    Returns:
        The document row ids.
    """
    statement = (
        select(table.document_id).where(table.model == model, table.field == field).distinct()
    )

    return set(session.exec(statement))


def create_vector_index(session: Session, table: type[EmbeddingRecord]) -> None:
    """Build the StreamingDiskANN (pgvectorscale) cosine index of a table.

    Args:
        session: Active database session.
        table: Embedding table to index.
    """
    name = table.__tablename__
    connection = session.connection()
    connection.execute(text(f"SET maintenance_work_mem = '{INDEX_MEMORY}'"))
    connection.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS {name}_embedding_idx "
            f"ON {name} USING diskann (embedding vector_cosine_ops)"
        )
    )
    session.commit()


def drop_vector_index(session: Session, table: type[EmbeddingRecord]) -> None:
    """Drop a table's vector index, so a large load does not update it row by row.

    Args:
        session: Active database session.
        table: Embedding table.
    """
    session.connection().execute(text(f"DROP INDEX IF EXISTS {table.__tablename__}_embedding_idx"))
    session.commit()


def embedded_sources(session: Session, dimensions: int, model: str, field: str) -> list[SourceType]:
    """List the sources that have embeddings of a field from a model.

    Args:
        session: Active database session.
        dimensions: Vector size of the model.
        model: Embedding model name.
        field: Embedded field.

    Returns:
        The sources with at least one row.
    """
    return [
        source_type
        for source_type, table in EMBEDDING_TABLES[dimensions].items()
        if session.exec(
            select(table.id).where(table.model == model, table.field == field).limit(1)
        ).first()
    ]


def search_embeddings(
    session: Session,
    table: type[EmbeddingRecord],
    source_type: SourceType,
    model: str,
    field: str,
    vector: Sequence[float],
    limit: int,
) -> list[SearchHit]:
    """Find the texts closest to a query vector in one source's table.

    Args:
        session: Active database session.
        table: The source's embedding table.
        source_type: Source the table holds.
        model: Embedding model the query vector comes from.
        field: Embedded field to search.
        vector: Query vector.
        limit: Number of hits.

    Returns:
        The closest texts, closest first.
    """
    distance = table.embedding.cosine_distance(vector)  # type: ignore[attr-defined]
    statement = (
        select(table, distance.label("distance"))
        .where(table.model == model, table.field == field)
        .order_by(distance)
        .limit(limit)
    )

    return [
        SearchHit(
            doc_id=record.doc_id,
            document_id=record.document_id,
            source_type=source_type,
            field=record.field,
            position=record.position,
            text=record.text,
            distance=float(row_distance),
        )
        for record, row_distance in session.exec(statement)
    ]


def embedded_doc_ids(session: Session, dimensions: int, model: str, field: str) -> set[str]:
    """Collect the dataset ids of every document with embeddings of a field from a model.

    Args:
        session: Active database session.
        dimensions: Vector size of the model.
        model: Embedding model name.
        field: Embedded field.

    Returns:
        The doc ids across all embedding tables of that size.
    """
    return {
        doc_id
        for table in EMBEDDING_TABLES[dimensions].values()
        for doc_id in session.exec(
            select(table.doc_id).where(table.model == model, table.field == field).distinct()
        )
    }


def pending_documents(
    session: Session,
    dataset: str,
    source_type: SourceType,
    table: type[EmbeddingRecord],
    model: str,
    field: str,
    selected: set[int] | None = None,
) -> list[tuple[int, int]]:
    """Find one source's documents that have no embeddings of a field from a model yet.

    Args:
        session: Active database session.
        dataset: Dataset to check.
        source_type: Source to check.
        table: The source's embedding table.
        model: Embedding model name.
        field: Embedded field.
        selected: Only these documents, or None for all.

    Returns:
        (row id, characters) of each pending document, in row order.
    """
    done = embedded_document_ids(session, table, model, field)

    return [
        (document_id, chars)
        for document_id, chars in document_lengths(session, dataset, source_type)
        if document_id not in done and (selected is None or document_id in selected)
    ]


def store_chunks(
    session: Session,
    table: type[EmbeddingRecord],
    model: str,
    field: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
) -> None:
    """Insert chunks with their vectors, each linked to its neighbours, without committing.

    Row ids are reserved first, so every chunk stores the ids of the chunks before and
    after it in its document (`previous_id`, `next_id`) in the same insert.

    Args:
        session: Active database session.
        table: Embedding table of the chunks' source.
        model: Embedding model name.
        field: Embedded field, e.g. `chunk`.
        chunks: Chunks in document order.
        vectors: One vector per chunk.
    """
    ids = reserve_ids(session, table, len(chunks))
    neighbours = neighbour_ids([chunk.document_id for chunk in chunks], ids)
    records = [
        table(
            id=row_id,
            document_id=chunk.document_id,
            doc_id=chunk.doc_id,
            field=field,
            position=chunk.position,
            previous_id=previous_id,
            next_id=next_id,
            model=model,
            text=chunk.text,
            embedding=list(vector),
        )
        for chunk, vector, row_id, (previous_id, next_id) in zip(
            chunks, vectors, ids, neighbours, strict=True
        )
    ]
    insert_embeddings(session, table, records)


def neighbour_ids(owners: Sequence[int], ids: Sequence[int]) -> list[tuple[int | None, int | None]]:
    """Link each chunk to the chunks before and after it in the same document.

    Args:
        owners: The document of each chunk, chunks in document order.
        ids: The row id of each chunk.

    Returns:
        (previous id, next id) per chunk; None where the document starts or ends.
    """
    return [
        (
            ids[index - 1] if index > 0 and owners[index - 1] == owner else None,
            ids[index + 1] if index + 1 < len(owners) and owners[index + 1] == owner else None,
        )
        for index, owner in enumerate(owners)
    ]
