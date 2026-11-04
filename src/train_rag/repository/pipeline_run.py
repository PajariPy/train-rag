"""Queries for the pipeline runs table."""

from datetime import UTC, datetime, timedelta

from sqlmodel import Session, col, select

from train_rag.models.pipeline_run import RunState
from train_rag.tables import PipelineRunRecord

# A running pipeline updates its row after every batch, at least every few seconds; a row
# untouched for this long belongs to a run that was stopped or crashed.
STALE_AFTER = timedelta(minutes=5)


def start_run(
    session: Session, pipeline: str, total_items: int, total_chars: int, unit: str
) -> PipelineRunRecord:
    """Record the start of a run.

    Args:
        session: Active database session.
        pipeline: Name of the pipeline and its settings, e.g. `data-ingestion gte`.
        total_items: Items the run has to process.
        total_chars: Their characters, or 0 to not weigh items by size.
        unit: What the items are, e.g. `documents` or `questions`.

    Returns:
        The stored run.
    """
    now = datetime.now(UTC)
    run = PipelineRunRecord(
        pipeline=pipeline,
        stage="starting",
        status="running",
        unit=unit,
        total_items=total_items,
        total_chars=total_chars,
        started_at=now,
        updated_at=now,
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    return run


def update_run(
    session: Session,
    run: PipelineRunRecord,
    stage: str,
    items: int,
    chars: int,
    status: str = "running",
    note: str | None = None,
) -> None:
    """Add a finished batch to a run.

    Args:
        session: Active database session.
        run: The run, from `start_run`.
        stage: What the run is working on now.
        items: Items finished since the last update.
        chars: Their characters.
        status: `running`, or `finished` for the last update.
        note: New extra line for the page, or None to keep the current one.
    """
    run.stage = stage
    run.status = status
    run.done_items += items
    run.done_chars += chars
    if note is not None:
        run.note = note
    run.updated_at = datetime.now(UTC)
    session.add(run)
    session.commit()


def latest_run_state(session: Session) -> RunState | None:
    """Describe the most recent run, with its elapsed and estimated remaining time.

    The time left assumes the run keeps its average speed, in characters per second when
    the run counts characters and in items per second otherwise.

    Args:
        session: Active database session.

    Returns:
        The run's state, or None when no run was recorded.
    """
    run = session.exec(
        select(PipelineRunRecord).order_by(col(PipelineRunRecord.id).desc()).limit(1)
    ).first()
    if run is None or run.id is None:
        return None

    now = datetime.now(UTC)
    status = run.status
    if status == "running" and now - run.updated_at > STALE_AFTER:
        status = "stopped"
    end = now if status == "running" else run.updated_at
    elapsed = (end - run.started_at).total_seconds()
    done, total = (
        (run.done_chars, run.total_chars) if run.total_chars else (run.done_items, run.total_items)
    )
    speed = done / elapsed if elapsed > 0 else 0.0
    seconds_left = None
    if status == "finished":
        seconds_left = 0.0
    elif status == "running" and speed > 0:
        seconds_left = (total - done) / speed

    return RunState(
        id=run.id,
        pipeline=run.pipeline,
        stage=run.stage,
        status=status,
        unit=run.unit,
        done_items=run.done_items,
        total_items=run.total_items,
        done_chars=run.done_chars,
        total_chars=run.total_chars,
        note=run.note,
        started_at=run.started_at,
        updated_at=run.updated_at,
        seconds_elapsed=elapsed,
        seconds_left=seconds_left,
    )
