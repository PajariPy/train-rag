"""Parsed Fireflies meeting transcripts."""

from datetime import date

from pydantic import Field

from train_rag.models.parsed.base import ParsedDocument, Person


class FirefliesMeeting(ParsedDocument):
    """A meeting with the summary and header data Fireflies already provides."""

    summary: str | None = None
    topics: list[str] = Field(default_factory=list)
    attendees: list[Person] = Field(default_factory=list)
    held_on: date | None = None
    start_time: str | None = None
    duration: str | None = None
