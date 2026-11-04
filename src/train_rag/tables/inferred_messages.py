"""Slack messages split by an LLM, stored apart from the raw documents."""

from typing import Any

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class InferredMessagesRecord(SQLModel, table=True):
    """The messages of one Slack thread, as split by an LLM from its text.

    `messages` holds `SlackMessage` objects (speaker, optional role and text). `method`
    names the tool that produced them, so a rerun can replace its own rows.
    """

    __tablename__ = "inferred_messages"

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", unique=True, index=True)
    messages: list[dict[str, Any]] = Field(sa_column=Column(JSONB, nullable=False))
    method: str = Field(index=True)
