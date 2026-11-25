"""Field search's own table: one row per mined document, its profiles and its filters.

Built by `train-rag index-fields` from `document_fields` and `document_field_vectors`: the
preferred extractor's row of each document and its three profile vectors. Each vector
column has a StreamingDiskANN cosine index, so a field search is an indexed query, and the
filters (areas, source, channel) sit in the same row.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlmodel import Field, SQLModel

from train_rag.tables.field_vector import PROFILE_DIMENSIONS


class FieldSearchRecord(SQLModel, table=True):
    """One mined document as field search finds it, for one embedding model."""

    __tablename__ = "field_search_vectors"

    document_fields_id: int = Field(foreign_key="document_fields.id", primary_key=True)
    model: str = Field(primary_key=True)
    document_id: int = Field(index=True)
    doc_id: str = Field(index=True)
    source_type: str
    channel: str = ""
    areas: list[str] = Field(sa_column=Column(ARRAY(Text), nullable=False))
    subject: list[float] | None = Field(
        default=None, sa_column=Column(Vector(PROFILE_DIMENSIONS), nullable=True)
    )
    problem: list[float] | None = Field(
        default=None, sa_column=Column(Vector(PROFILE_DIMENSIONS), nullable=True)
    )
    names: list[float] | None = Field(
        default=None, sa_column=Column(Vector(PROFILE_DIMENSIONS), nullable=True)
    )
