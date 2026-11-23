"""The report of an evaluation with clarifying questions, and the helpers of its commands."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from train_rag.evaluation.agent_tools import ASK_TOOL
from train_rag.models.memory import MemoryNote


def read_question_ids(path: Path) -> list[str]:
    """Read question ids from a file: one per line, `#` comments and blank lines skipped.

    Args:
        path: The file.

    Returns:
        The ids, in file order.
    """
    lines = (line.strip() for line in path.read_text().splitlines())

    return [line for line in lines if line and not line.startswith("#")]


def missing_memories(question_ids: Sequence[str], memories: Mapping[str, MemoryNote]) -> list[str]:
    """List the questions that have no memory note.

    Args:
        question_ids: Questions to check.
        memories: Stored notes keyed by question id.

    Returns:
        The ids without a note, in the given order.
    """
    return [question_id for question_id in question_ids if question_id not in memories]


def count_asks(trace: Sequence[Mapping[str, Any]]) -> int:
    """Count the questions an agent put to the person in one answer's trace.

    Args:
        trace: The stored trace steps.

    Returns:
        The number of `ask_user` calls.
    """
    return sum(step.get("tool") == ASK_TOOL for step in trace)


def row(name: str, correct: Sequence[bool | None]) -> str:
    """One table row: questions, correct answers and accuracy; ungraded counts as wrong.

    Args:
        name: Row label.
        correct: Each question's verdict.

    Returns:
        The Markdown row.
    """
    right = sum(verdict is True for verdict in correct)
    accuracy = 100 * right / len(correct) if correct else 0.0

    return f"| {name} | {len(correct)} | {right} | {accuracy:.0f}% |"


def clarify_report(
    baseline: Mapping[str, bool | None],
    run: Mapping[str, bool | None],
    asks: Mapping[str, int],
    memories: Mapping[str, MemoryNote],
) -> str:
    """Compare zero-shot and clarified accuracy, split by clean and misremembered notes.

    Args:
        baseline: Each question's verdict in the zero-shot run.
        run: Each question's verdict in the run with clarifying questions.
        asks: Questions the agent asked, per question (missing means none).
        memories: The notes, to tell clean ones from ones with a wrong detail.

    Returns:
        A Markdown table and two lines on how often the agent asked.
    """
    ids = sorted(run)
    clean = [question_id for question_id in ids if memories[question_id].clean]
    wrong = [question_id for question_id in ids if not memories[question_id].clean]
    asked = [asks.get(question_id, 0) for question_id in ids]
    lines = [
        "| | Questions | Correct | Accuracy |",
        "|---|---:|---:|---:|",
        row("Zero-shot", [baseline.get(question_id) for question_id in ids]),
        row("With clarification", [run[question_id] for question_id in ids]),
        row("— clean notes", [run[question_id] for question_id in clean]),
        row("— note with a wrong detail", [run[question_id] for question_id in wrong]),
        "",
        f"asked at least once: {sum(count > 0 for count in asked)} of {len(ids)}",
        f"average asks: {sum(asked) / len(ids) if ids else 0.0:.1f}",
    ]

    return "\n".join(lines)
