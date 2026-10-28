"""Parsed Jira tickets."""

from datetime import datetime

from pydantic import Field

from train_rag.models.parsed.base import Comment, ParsedDocument


class JiraTicket(ParsedDocument):
    """A Jira ticket; `issue_summary` is used as its summary for now."""

    issue_summary: str | None = None
    description: str | None = None
    impact: str | None = None
    environment: str | None = None
    comments: list[Comment] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    first_activity_at: datetime | None = None
