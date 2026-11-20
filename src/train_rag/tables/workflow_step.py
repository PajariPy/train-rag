"""Workflow steps: a multi-step job shown on the progress page like CI/CD steps."""

from datetime import datetime

from sqlalchemy import Column, DateTime
from sqlmodel import Field, SQLModel


class WorkflowStepRecord(SQLModel, table=True):
    """One step of a workflow, such as "Full training" in an SLM fine-tuning job.

    A workflow is the set of steps sharing a `workflow` key, ordered by `position`. A step
    is `queued`, `running`, `success`, `failed` or `skipped`. `run_id` points to the
    pipeline run that records the step's own progress bar, when it has one; a step
    reported by a shell script (renting a GPU, copying files) may have none.
    """

    __tablename__ = "workflow_steps"

    id: int | None = Field(default=None, primary_key=True)
    workflow: str = Field(index=True)
    position: int
    name: str
    status: str = "queued"
    run_id: int | None = None
    note: str = ""
    created_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    started_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    finished_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
