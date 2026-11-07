"""Queries for evaluation runs and their results."""

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlmodel import Session, col, select

from train_rag.models.evaluation import QuestionResult, RunSummary
from train_rag.tables import EvalResultRecord, EvalRunRecord


def create_run(
    session: Session, dataset: str, system: str, judge: str, question_count: int
) -> EvalRunRecord:
    """Store a new, unfinished run.

    Args:
        session: Active database session.
        dataset: Dataset the questions come from.
        system: Name of the evaluated RAG system.
        judge: Name of the answer judge.
        question_count: Number of questions in the run.

    Returns:
        The stored run with its id.
    """
    run = EvalRunRecord(dataset=dataset, system=system, judge=judge, question_count=question_count)
    session.add(run)
    session.commit()
    session.refresh(run)

    return run


def add_results(session: Session, run_id: int, results: Sequence[QuestionResult]) -> None:
    """Store per-question results of a run.

    Args:
        session: Active database session.
        run_id: Run the results belong to.
        results: Evaluated questions.
    """
    session.add_all(
        EvalResultRecord(
            run_id=run_id,
            score=result.score,
            **result.model_dump(exclude={"question_type"}),
            question_type=str(result.question_type),
        )
        for result in results
    )
    session.commit()


def finish_run(session: Session, run: EvalRunRecord, summary: RunSummary) -> None:
    """Mark a run finished and store its summary.

    Args:
        session: Active database session.
        run: The run.
        summary: Its metrics, overall and per question type.
    """
    run.finished_at = datetime.now(UTC)
    run.summary = summary.model_dump(mode="json")
    session.add(run)
    session.commit()


def list_runs(session: Session, dataset: str) -> list[EvalRunRecord]:
    """List runs, newest first.

    Args:
        session: Active database session.
        dataset: Dataset the runs evaluated.

    Returns:
        The runs.
    """
    statement = (
        select(EvalRunRecord)
        .where(EvalRunRecord.dataset == dataset)
        .order_by(col(EvalRunRecord.id).desc())
    )

    return list(session.exec(statement))


def results_of_run(session: Session, run_id: int) -> list[EvalResultRecord]:
    """Load every per-question result of a run.

    Args:
        session: Active database session.
        run_id: The run.

    Returns:
        The results ordered by question id.
    """
    statement = (
        select(EvalResultRecord)
        .where(EvalResultRecord.run_id == run_id)
        .order_by(col(EvalResultRecord.question_id))
    )

    return list(session.exec(statement))


def wrong_question_ids(session: Session, run_id: int) -> set[str]:
    """List the questions a run answered wrong or failed on.

    Args:
        session: Active database session.
        run_id: The run.

    Returns:
        Ids of the questions not judged correct.
    """
    statement = select(EvalResultRecord.question_id).where(
        EvalResultRecord.run_id == run_id, col(EvalResultRecord.correct).is_not(True)
    )

    return set(session.exec(statement))
