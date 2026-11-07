"""Async queries of the retrieval pipeline: nearest chunks, and the texts and titles of hits."""

from collections.abc import Sequence

from sqlalchemy import tuple_
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from train_rag.models import SourceType
from train_rag.models.retrieval import SearchHit
from train_rag.tables import DocumentRecord, EmbeddingRecord


async def nearest_chunks(
    session: AsyncSession,
    table: type[EmbeddingRecord],
    source_type: SourceType,
    model: str,
    field: str,
    vector: Sequence[float],
    limit: int,
) -> list[SearchHit]:
    """Find the chunks closest to a query vector in one source's table, without their text.

    Chunks can be tens of thousands of characters long, so texts are loaded only for the
    final hits, with `chunk_texts`.

    Args:
        session: Active async database session.
        table: The source's embedding table.
        source_type: Source the table holds.
        model: Embedding model the query vector comes from.
        field: Embedded field to search.
        vector: Query vector.
        limit: Number of chunks.

    Returns:
        The closest chunks, closest first, with empty text.
    """
    distance = table.embedding.cosine_distance(vector)  # type: ignore[attr-defined]
    statement = (
        select(table.document_id, table.doc_id, table.position, distance.label("distance"))
        .where(table.model == model, table.field == field)
        .order_by(distance)
        .limit(limit)
    )
    rows = await session.exec(statement)

    return [
        SearchHit(
            doc_id=doc_id,
            document_id=document_id,
            source_type=source_type,
            field=field,
            position=position,
            text="",
            distance=float(row_distance),
        )
        for document_id, doc_id, position, row_distance in rows
    ]


async def chunk_texts(
    session: AsyncSession,
    table: type[EmbeddingRecord],
    model: str,
    field: str,
    keys: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], str]:
    """Load the text of chunks by document and position.

    Args:
        session: Active async database session.
        table: The source's embedding table.
        model: Embedding model of the chunks.
        field: Embedded field.
        keys: (document row id, chunk position) of each chunk.

    Returns:
        Chunk texts keyed by (document row id, position); missing chunks are left out.
    """
    if not keys:
        return {}

    statement = select(table.document_id, table.position, table.text).where(
        table.model == model,
        table.field == field,
        tuple_(col(table.document_id), col(table.position)).in_(keys),
    )
    rows = await session.exec(statement)

    return {(document_id, position): text for document_id, position, text in rows}


async def document_titles(session: AsyncSession, ids: Sequence[int]) -> dict[int, str]:
    """Load document titles by row id.

    Args:
        session: Active async database session.
        ids: Document row ids.

    Returns:
        Titles keyed by row id.
    """
    statement = select(DocumentRecord.id, DocumentRecord.title).where(
        col(DocumentRecord.id).in_(ids)
    )
    rows = await session.exec(statement)

    return {document_id: title for document_id, title in rows if document_id is not None}
