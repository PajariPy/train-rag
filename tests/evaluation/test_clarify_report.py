"""Tests for the clarification report and the helpers of its commands."""

from pathlib import Path

from train_rag.evaluation.clarify_report import (
    clarify_report,
    count_asks,
    missing_memories,
    read_question_ids,
)
from train_rag.models.memory import MemoryNote

MEMORIES = {
    "a": MemoryNote(question_id="a", note="n"),
    "b": MemoryNote(question_id="b", note="n", wrong_detail="w"),
}


def test_report_splits_clean_and_wrong() -> None:
    """Rows for zero-shot, with clarification, clean notes and wrong notes."""
    text = clarify_report(
        {"a": False, "b": True}, {"a": True, "b": False}, {"a": 2, "b": 0}, MEMORIES
    )
    lines = {line.split("|")[1].strip(): line for line in text.splitlines() if "|" in line}

    assert "1 | 50%" in lines["Zero-shot"]
    assert "1 | 50%" in lines["With clarification"]
    assert "| 1 | 1 | 100%" in lines["— clean notes"]
    assert "| 1 | 0 | 0%" in lines["— note with a wrong detail"]
    assert "asked at least once: 1 of 2" in text
    assert "average asks: 1.0" in text


def test_report_counts_ungraded_as_wrong() -> None:
    """An ungraded answer (None) doesn't count as correct."""
    text = clarify_report({"a": None, "b": None}, {"a": None, "b": True}, {}, MEMORIES)

    assert "With clarification | 2 | 1 | 50%" in text


def test_count_asks() -> None:
    """Only ask_user tool steps count."""
    trace = [{"kind": "tool", "tool": "ask_user"}, {"kind": "tool", "tool": "keyword_search"}]

    assert count_asks(trace + trace[:1]) == 2


def test_missing_memories_listed() -> None:
    """Questions without a note are listed in order."""
    assert missing_memories(["a", "c", "b", "d"], MEMORIES) == ["c", "d"]


def test_read_question_ids_skips_blanks_and_comments(tmp_path: Path) -> None:
    """One id per line; blank lines and # comments are skipped, spaces stripped."""
    path = tmp_path / "ids.txt"
    path.write_text("# failures\nqst_1\n\n qst_2 \n")

    assert read_question_ids(path) == ["qst_1", "qst_2"]
