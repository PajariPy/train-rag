"""Queries for workflows: multi-step jobs shown on the progress page like CI/CD steps."""

from datetime import UTC, datetime

from sqlmodel import Session, col, select

from train_rag.models.workflow import StepState, WorkflowState
from train_rag.repository.pipeline_run import run_state, set_run_progress, start_run
from train_rag.tables import PipelineRunRecord, WorkflowStepRecord

STATUSES = ("queued", "running", "success", "failed", "skipped")
DONE = ("success", "failed", "skipped")


def start_workflow(session: Session, name: str, steps: list[str]) -> str:
    """Record a new workflow with all its steps queued.

    Args:
        session: Active database session.
        name: What the workflow does, e.g. `SLM fine-tuning`.
        steps: The step names, in order.

    Returns:
        The workflow's key: its name and start time, used to report its steps.
    """
    now = datetime.now(UTC)
    key = f"{name} · {now:%Y-%m-%d %H:%M}"
    for position, step in enumerate(steps):
        session.add(
            WorkflowStepRecord(workflow=key, position=position, name=step, created_at=now)
        )
    session.commit()

    return key


def find_step(session: Session, workflow: str, step: str) -> WorkflowStepRecord:
    """Look up one step of a workflow.

    Args:
        session: Active database session.
        workflow: The workflow's key, from `start_workflow`.
        step: The step's name.

    Returns:
        The step.

    Raises:
        KeyError: If the workflow has no such step.
    """
    record = session.exec(
        select(WorkflowStepRecord).where(
            WorkflowStepRecord.workflow == workflow, WorkflowStepRecord.name == step
        )
    ).first()
    if record is None:
        raise KeyError(f"no step {step!r} in workflow {workflow!r}")

    return record


def set_step(
    session: Session,
    workflow: str,
    step: str,
    status: str,
    note: str | None = None,
    run_id: int | None = None,
) -> None:
    """Change a step's status; the first `running` sets its start, a final status its end.

    Setting a step back to `queued` (a retry) clears its times and its progress bar.

    Args:
        session: Active database session.
        workflow: The workflow's key.
        step: The step's name.
        status: One of STATUSES.
        note: New extra line for the step, or None to keep the current one.
        run_id: The pipeline run that records the step's progress, or None to keep it.

    Raises:
        ValueError: If the status is unknown.
    """
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; use one of {STATUSES}")
    record = find_step(session, workflow, step)
    now = datetime.now(UTC)
    if status == "queued":  # queued again for a retry: the next run starts fresh
        record.started_at = record.finished_at = None
        record.run_id = None
    if status == "running" and record.started_at is None:
        record.started_at = now
    if status in DONE:
        record.finished_at = now
        record.started_at = record.started_at or now
    record.status = status
    if note is not None:
        record.note = note
    if run_id is not None:
        record.run_id = run_id
    session.add(record)
    session.commit()


def report_step_progress(
    session: Session,
    workflow: str,
    step: str,
    done: int,
    total: int,
    unit: str,
    note: str | None = None,
) -> None:
    """Set a step's progress bar from outside Python (a shell script polling a GPU).

    Creates the step's pipeline run on the first report; later reports set the absolute
    numbers. Marks the step running.

    Args:
        session: Active database session.
        workflow: The workflow's key.
        step: The step's name.
        done: Items finished so far (e.g. training steps).
        total: Items in all.
        unit: What the items are, e.g. `training steps`.
        note: Extra line for the page, or None to keep the current one.
    """
    record = find_step(session, workflow, step)
    run = session.get(PipelineRunRecord, record.run_id) if record.run_id else None
    if run is None:
        run = start_run(session, step, total, 0, unit)
    set_run_progress(session, run, done, total, note)
    set_step(session, workflow, step, "running", note, run.id)


def overall_status(statuses: list[str]) -> str:
    """The workflow's status from its steps' statuses.

    Args:
        statuses: Every step's status.

    Returns:
        `failed` if a step failed, `running` while any step is queued or running,
        otherwise `success`.
    """
    if "failed" in statuses:
        return "failed"
    if any(status in ("queued", "running") for status in statuses):
        return "running"

    return "success"


def latest_workflow_state(session: Session) -> WorkflowState | None:
    """Describe the most recently started workflow and its steps.

    Args:
        session: Active database session.

    Returns:
        The workflow's state, or None when no workflow was recorded.
    """
    last = session.exec(
        select(WorkflowStepRecord).order_by(col(WorkflowStepRecord.id).desc()).limit(1)
    ).first()
    if last is None:
        return None
    records = session.exec(
        select(WorkflowStepRecord)
        .where(WorkflowStepRecord.workflow == last.workflow)
        .order_by(col(WorkflowStepRecord.position))
    ).all()

    now = datetime.now(UTC)
    steps = []
    for record in records:
        run = session.get(PipelineRunRecord, record.run_id) if record.run_id else None
        state = run_state(run) if run is not None else None
        seconds = None
        if record.started_at is not None:
            seconds = ((record.finished_at or now) - record.started_at).total_seconds()
        steps.append(
            StepState(
                name=record.name, status=record.status, note=record.note,
                seconds=seconds, run=state,
            )
        )
    status = overall_status([step.status for step in steps])
    started = min(record.created_at for record in records)
    finished = [r.finished_at for r in records if r.finished_at is not None]
    end = max(finished) if status != "running" and finished else now

    return WorkflowState(
        workflow=last.workflow,
        status=status,
        started_at=started,
        seconds_elapsed=(end - started).total_seconds(),
        steps=steps,
    )
