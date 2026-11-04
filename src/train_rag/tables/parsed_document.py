"""Parsed documents: the typed fields every parser found, stored next to the raw text."""

from typing import Any

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class ParsedDocumentRecord(SQLModel, table=True):
    """The parser's output for one raw document.

    `kind` is the parsed model's name (`SlackThread`, `OpenText`, …); `data` is the
    model dumped to JSON (participants, dates, channel and so on), for filtering and
    linking documents later. Sources without a parser (Google Drive, Confluence) have no
    row. `useful` is False for documents with too little text to index.
    """

    __tablename__ = "parsed_documents"

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", unique=True, index=True)
    doc_id: str = Field(index=True)
    source_type: str = Field(index=True)
    kind: str = Field(index=True)
    useful: bool
    flagged: bool
    data: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
