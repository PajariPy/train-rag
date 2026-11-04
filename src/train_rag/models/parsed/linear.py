"""Parsed Linear tickets."""

from datetime import datetime

from pydantic import Field

from train_rag.models.parsed.base import Comment, ParsedDocument


class LinearTicket(ParsedDocument):
    """A Linear ticket with its description and comment activity."""

    description: str | None = None
    comments: list[Comment] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    first_activity_at: datetime | None = None
