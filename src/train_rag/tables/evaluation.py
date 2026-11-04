"""Evaluation runs and their per-question results stored in Postgres."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Column, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class EvalRunRecord(SQLModel, table=True):
    """One evaluation of a RAG system on the benchmark questions.

    `summary` holds the run's `RunSummary` once it finishes.
    """

    __tablename__ = "eval_runs"

    id: int | None = Field(default=None, primary_key=True)
    dataset: str = Field(index=True)
    system: str = Field(index=True)
    judge: str
    question_count: int
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    finished_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    summary: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB, nullable=True))


class EvalResultRecord(SQLModel, table=True):
    """The evaluation of one question within a run."""

    __tablename__ = "eval_results"

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="eval_runs.id", index=True)
    question_id: str = Field(index=True)
    question_type: str = Field(index=True)
    answer: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    retrieved_doc_ids: list[str] = Field(sa_column=Column(JSONB, nullable=False))
    # The retrieved documents the answer says it is taken from, for systems that cite.
    cited_doc_ids: list[str] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=False, server_default="[]")
    )
    document_recall: float | None = None
    extra_docs: int
    correct: bool | None = None
    completeness: float | None = None
    score: float | None = None
    facts_covered: list[bool] = Field(sa_column=Column(JSONB, nullable=False))
    judge_reasoning: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    error: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # The steps the system took (see `TraceStep`), for systems that record them.
    trace: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=False, server_default="[]")
    )
