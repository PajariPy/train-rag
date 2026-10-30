"""Parsed Fireflies meeting transcripts."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from train_rag.models.parsed.base import ParsedDocument, Person


class InferredDate(BaseModel):
    """A meeting date estimated by an LLM judge from clues in the text, not read from a header.

    `precision` says how close it is: "day" when the text pins the day, "week" for
    estimates such as shortly before the earliest action item due date. When
    `year_in_evidence` is False the quoted clue has no year, so the year may be a guess.
    """

    held_on: date
    precision: Literal["day", "week", "month"]
    evidence: str
    year_in_evidence: bool


class FirefliesMeeting(ParsedDocument):
    """A meeting with the summary and header data Fireflies already provides.

    `held_on` is the date read from the header. When there is none, `held_on_inferred`
    holds the date the LLM judge estimated from the text, if it found one. Likewise
    `attendees_inferred` holds the speakers an LLM listed when the text has no attendee
    list or speaker lines the parser recognizes.
    """

    summary: str | None = None
    topics: list[str] = Field(default_factory=list)
    attendees: list[Person] = Field(default_factory=list)
    attendees_inferred: list[Person] = Field(default_factory=list)
    held_on: date | None = None
    held_on_inferred: InferredDate | None = None
    start_time: str | None = None
    duration: str | None = None
