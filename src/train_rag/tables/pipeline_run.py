"""Pipeline runs: what a run has to do and how far it got, for the progress page."""

from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime
from sqlmodel import Field, SQLModel


class PipelineRunRecord(SQLModel, table=True):
    """One run of a pipeline or an evaluation, updated after every batch.

    `total_items` is what this run had left when it started, counted in `unit` (e.g.
    documents or questions); `done_items` counts what it has finished since. The
    characters are optional (0 when a run doesn't weigh its items by size); when set, the
    time left is estimated from them. `stage` names the step it is on, and `note` holds
    any extra line for the page, such as the money spent so far. `status` is `running`
    until the run ends; a run that was killed stays `running` but stops updating
    `updated_at`.
    """

    __tablename__ = "pipeline_runs"

    id: int | None = Field(default=None, primary_key=True)
    pipeline: str = Field(index=True)
    stage: str
    status: str
    unit: str = "documents"
    total_items: int
    total_chars: int = Field(default=0, sa_type=BigInteger)
    done_items: int = 0
    done_chars: int = Field(default=0, sa_type=BigInteger)
    note: str = ""
    started_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    updated_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
