"""Embedding tables: one per source type and vector size, all with the same columns."""

import types
from typing import cast

from pgvector.sqlalchemy import Vector
from sqlalchemy import Text
from sqlmodel import Field, SQLModel

from train_rag.models import SourceType


class EmbeddingRecord(SQLModel):
    """One embedded text of a document, with its vector.

    A document can have many rows: per embedded field (`field`, e.g. `summary` or
    `chunk`), per chunk of that field (`position`, 0 for a whole field) and per embedding
    model (`model`). `previous_id` and `next_id` are the row ids of the neighbouring
    chunks of the same document, None at its first and last chunk. `doc_id` is the
    dataset id the benchmark questions cite; `document_id` links back to the raw document
    row. The vector is searched with a pgvectorscale StreamingDiskANN index on cosine
    distance.
    """

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", index=True)
    doc_id: str = Field(index=True)
    field: str = Field(index=True)
    position: int = 0
    previous_id: int | None = None
    next_id: int | None = None
    model: str = Field(index=True)
    text: str = Field(sa_type=Text)
    embedding: list[float]


class MiniEmbeddingRecord(EmbeddingRecord):
    """Columns of the tables for 384-dimensional vectors (MiniLM)."""

    embedding: list[float] = Field(sa_type=Vector(384))


class BaseEmbeddingRecord(EmbeddingRecord):
    """Columns of the tables for 768-dimensional vectors (base-size models)."""

    embedding: list[float] = Field(sa_type=Vector(768))


BASES: dict[int, tuple[type[EmbeddingRecord], str]] = {
    384: (MiniEmbeddingRecord, "embeddings"),
    768: (BaseEmbeddingRecord, "embeddings_768"),
}


# A fine-tuned model's vectors live in tables of their own, so the base model's index and
# rows stay untouched and the two can be compared side by side.
FINETUNED_SUFFIX = "embeddings_768_ft"


def source_table(
    dimensions: int, source_type: SourceType, finetuned: bool = False
) -> type[EmbeddingRecord]:
    """Define the table of one source for vectors of one size.

    Args:
        dimensions: Vector size.
        source_type: Source the table holds.
        finetuned: The fine-tuned model's table instead of the base model's.

    Returns:
        The table model, e.g. `slack_embeddings`, `slack_embeddings_768` or
        `slack_embeddings_768_ft`.
    """
    base, suffix = BASES[dimensions]
    if finetuned:
        suffix = FINETUNED_SUFFIX
    kind = "Finetuned" if finetuned else ""
    name = f"{source_type.value.title().replace('_', '')}{kind}{base.__name__}"
    table = types.new_class(
        name,
        (base,),
        {"table": True},
        lambda namespace: namespace.update(
            __tablename__=f"{source_type.value}_{suffix}",
            __doc__=f"{dimensions}-dimensional embeddings of {source_type.value} documents.",
            __module__=__name__,
        ),
    )

    return cast(type[EmbeddingRecord], table)


EMBEDDING_TABLES: dict[int, dict[SourceType, type[EmbeddingRecord]]] = {
    dimensions: {source_type: source_table(dimensions, source_type) for source_type in SourceType}
    for dimensions in BASES
}

FINETUNED_TABLES: dict[SourceType, type[EmbeddingRecord]] = {
    source_type: source_table(768, source_type, finetuned=True) for source_type in SourceType
}


def tables_for(dimensions: int, finetuned: bool) -> dict[SourceType, type[EmbeddingRecord]]:
    """The embedding tables an embedder reads and writes.

    Args:
        dimensions: The embedder's vector size.
        finetuned: Whether it is a fine-tuned model (it has tables of its own).

    Returns:
        Source -> table.
    """
    return FINETUNED_TABLES if finetuned else EMBEDDING_TABLES[dimensions]
