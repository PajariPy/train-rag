"""Load the title and chunk text of documents found by the retrieval pipeline."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from train_rag.models import SourceType
from train_rag.models.retrieval import RetrievedDocument
from train_rag.repository.search import chunk_texts, document_titles
from train_rag.tables import EmbeddingRecord


async def load_titles(engine: AsyncEngine, ids: list[int]) -> dict[int, str]:
    """Load document titles.

    Args:
        engine: Async database engine.
        ids: Document row ids.

    Returns:
        Titles keyed by row id.
    """
    async with AsyncSession(engine) as session:
        return await document_titles(session, ids)


async def load_texts(
    engine: AsyncEngine,
    table: type[EmbeddingRecord],
    model: str,
    field: str,
    keys: list[tuple[int, int]],
) -> dict[tuple[int, int], str]:
    """Load the chunk texts of documents of one source.

    Args:
        engine: Async database engine.
        table: The source's embedding table.
        model: Embedding model of the chunks.
        field: Embedded field.
        keys: (document row id, chunk position) of each chunk.

    Returns:
        Chunk texts keyed by (document row id, position).
    """
    async with AsyncSession(engine) as session:
        return await chunk_texts(session, table, model, field, keys)


async def describe_documents(
    engine: AsyncEngine,
    tables: dict[SourceType, type[EmbeddingRecord]],
    model: str,
    field: str,
    documents: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    """Add each document's title and the text of its chunk.

    Args:
        engine: Async database engine.
        tables: Embedding table of each source.
        model: Embedding model of the chunks.
        field: Embedded field.
        documents: Documents with a chunk position.

    Returns:
        The same documents, in order, with title and text; text is empty for a chunk
        that isn't stored.
    """
    sources = {document.source_type for document in documents}
    keys_by_source = {
        source_type: [
            (document.document_id, document.position)
            for document in documents
            if document.source_type == source_type
        ]
        for source_type in sources
    }
    titles, texts = await asyncio.gather(
        load_titles(engine, [document.document_id for document in documents]),
        asyncio.gather(
            *(
                load_texts(engine, tables[source_type], model, field, keys)
                for source_type, keys in keys_by_source.items()
            )
        ),
    )
    text_of = {key: text for source_texts in texts for key, text in source_texts.items()}

    return [
        document.model_copy(
            update={
                "title": titles.get(document.document_id, ""),
                "text": text_of.get((document.document_id, document.position), ""),
            }
        )
        for document in documents
    ]
