"""The state of a workflow and its steps, as the progress page shows them."""

from datetime import datetime

from pydantic import BaseModel

from train_rag.models.pipeline_run import RunState


class StepState(BaseModel):
    """One step: its status, how long it has run, and its own progress when it has a run.

    `status` is `queued`, `running`, `success`, `failed` or `skipped`. `seconds` is None
    for a step that hasn't started.
    """

    name: str
    status: str
    note: str
    seconds: float | None
    run: RunState | None


class WorkflowState(BaseModel):
    """A workflow: its overall status and its steps in order.

    The overall status is `failed` if a step failed, `running` while any step is queued
    or running, and `success` when every step succeeded or was skipped.
    """

    workflow: str
    status: str
    started_at: datetime
    seconds_elapsed: float
    steps: list[StepState]
