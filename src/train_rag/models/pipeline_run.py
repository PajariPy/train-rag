"""The state of a pipeline run, as the progress page shows it."""

from datetime import datetime

from pydantic import BaseModel


class RunState(BaseModel):
    """How far a run got and how much is left.

    Items are counted in `unit` (e.g. documents or questions); `total_chars` is 0 for a
    run that doesn't weigh its items by size. `status` is `running`, `finished`, or
    `stopped` for a run that stopped updating without finishing. `seconds_left` is None
    until the run has a measurable speed.
    """

    id: int
    pipeline: str
    stage: str
    status: str
    unit: str
    done_items: int
    total_items: int
    done_chars: int
    total_chars: int
    note: str
    started_at: datetime
    updated_at: datetime
    seconds_elapsed: float
    seconds_left: float | None
