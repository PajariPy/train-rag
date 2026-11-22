"""Mined fields per document, one row per document and extractor."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Column, DateTime, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlmodel import Field, SQLModel


class DocumentFieldsRecord(SQLModel, table=True):
    """What one extractor mined from one document.

    `extractor` names the model and prompt (`deepseek-fc8ae8878988` for the DeepSeek labels,
    the SLM's run name later), so several extractors' rows sit side by side and search picks
    one per document by preference. `raw` is the line exactly as the extractor wrote it;
    `fields` holds the parsed values of every schema field (lists, maybe empty). `channel` is
    the Slack channel from the parsed document, copied here for filtering.
    """

    __tablename__ = "document_fields"
    __table_args__ = (UniqueConstraint("document_id", "extractor"),)

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", index=True)
    doc_id: str = Field(index=True)
    source_type: str = Field(index=True)
    extractor: str = Field(index=True)
    raw: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
    fields: dict[str, list[str]] = Field(sa_column=Column(JSONB, nullable=False))
    thread_kind: str | None = None
    status: str | None = None
    areas: list[str] = Field(sa_column=Column(ARRAY(Text), nullable=False))
    channel: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
