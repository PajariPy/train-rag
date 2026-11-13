"""Memory notes of benchmark questions, stored in Postgres."""

from datetime import UTC, datetime

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


class QuestionMemoryRecord(SQLModel, table=True):
    """What the person behind one question remembers, and how the note was written."""

    __tablename__ = "question_memories"

    question_id: str = Field(primary_key=True)
    note: str = Field(sa_column=Column(Text, nullable=False))
    wrong_detail: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    model: str
    prompt_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
