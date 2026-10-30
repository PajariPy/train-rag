"""Meeting attendees listed by an LLM, stored apart from the raw documents."""

from typing import Any

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class InferredAttendeesRecord(SQLModel, table=True):
    """The people who speak in one meeting, as listed by an LLM from its text.

    `attendees` holds `Person` objects (name and optional role). `method` names the
    tool that produced them, so a rerun can replace its own rows.
    """

    __tablename__ = "inferred_attendees"

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", unique=True, index=True)
    attendees: list[dict[str, Any]] = Field(sa_column=Column(JSONB, nullable=False))
    method: str = Field(index=True)
