"""Raw dataset documents stored in Postgres."""

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


class DocumentRecord(SQLModel, table=True):
    """One source document with its provenance and size.

    `doc_id` is not unique: the dataset reuses a few ids for different documents
    (near-duplicates with conflicting facts), so a row is identified by where it
    came from instead.
    """

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("dataset", "source_file", "row_index"),)

    id: int | None = Field(default=None, primary_key=True)
    doc_id: str = Field(index=True)
    dataset: str = Field(index=True)
    source_type: str = Field(index=True)
    title: str = Field(sa_column=Column(Text, nullable=False))
    content: str = Field(sa_column=Column(Text, nullable=False))
    char_count: int
    byte_size: int
    source_file: str
    row_index: int
    loaded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
