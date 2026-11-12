"""Tests for memory notes: which notes get a wrong detail, and the title-leak check."""

from types import SimpleNamespace
from typing import cast

import pytest
from google import genai
from google.genai import types

from train_rag.evaluation.memory_notes import (
    has_number,
    leaks_title,
    pick_wrong,
    write_each,
    write_note,
)
from train_rag.evaluation.usage import UsageMeter
from train_rag.models import Question, QuestionType
from train_rag.models.memory import MemoryNote
from train_rag.prompts import prompt

IDS = [f"qst_{n:04d}" for n in range(46)]


def test_pick_wrong_takes_a_quarter_reproducibly() -> None:
    """A quarter of 46 rounds to 12, the same whatever the input order."""
    first = pick_wrong(IDS)

    assert len(first) == 12
    assert first == pick_wrong(list(reversed(IDS)))
    assert first <= set(IDS)


def test_pick_wrong_seed_changes_choice() -> None:
    """Another seed picks other notes."""
    assert pick_wrong(IDS, seed=1) != pick_wrong(IDS, seed=2)


def test_leaks_title() -> None:
    """Two consecutive long title words in the note count as a leak; one alone doesn't."""
    title = "introduce-kernel-stability-thresholds-and-precision-annealing"

    assert leaks_title("- a PR about precision annealing", title)
    assert not leaks_title("- a PR about falling back to full precision", title)


def test_memory_note_clean() -> None:
    """A note is clean when it has no wrong detail."""
    assert MemoryNote(question_id="q", note="- a PR").clean
    assert not MemoryNote(
        question_id="q", note="- a Jira ticket", wrong_detail="says Jira; it is Linear"
    ).clean


class RecordingModels:
    """Replays prepared reply texts and records every request as one text."""

    def __init__(self, replies: list[str]) -> None:
        """Keep the replies to replay.

        Args:
            replies: One reply text per call, in order.
        """
        self.replies = replies
        self.requests: list[str] = []

    def generate_content(
        self, model: str, contents: object, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        """Record the request (instructions and contents) and return the next reply.

        Args:
            model: Ignored.
            contents: The request contents.
            config: The request settings.

        Returns:
            A response whose text is the next reply.
        """
        self.requests.append(f"{config.system_instruction}\n{contents}")
        text = self.replies.pop(0)

        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=[types.Part.from_text(text=text)])
                )
            ]
        )


def fake_client(replies: list[str]) -> tuple[genai.Client, RecordingModels]:
    """Build a Gemini client whose model calls are recorded and replayed.

    Args:
        replies: One reply text per call.

    Returns:
        The client and its recording models.
    """
    models = RecordingModels(replies)

    return cast(genai.Client, SimpleNamespace(models=models)), models


QUESTION = Question(
    id="qst_0178",
    question_type=QuestionType.SEMANTIC,
    text="What is the default pass rate for stepping down from the safest numeric mode?",
    gold_answer="GOLD-ANSWER-0.92",
    expected_doc_ids=["doc-1"],
)


def test_write_note_clean_drops_wrong_detail() -> None:
    """A clean note never keeps a wrong detail, whatever the model returns."""
    client, _ = fake_client(['{"note": "- a PR", "wrong_detail": "x"}'])

    note = write_note(client, UsageMeter(), QUESTION, "doc", wrong=False)

    assert note.wrong_detail is None
    assert note.note == "- a PR"
    assert note.question_id == "qst_0178"


def test_write_note_wrong_requires_detail() -> None:
    """A note picked for a wrong detail must say which detail is wrong."""
    client, _ = fake_client(['{"note": "- a PR", "wrong_detail": null}'])

    with pytest.raises(ValueError):
        write_note(client, UsageMeter(), QUESTION, "doc", wrong=True)


def test_write_note_sends_question_and_document() -> None:
    """The writer sees the question and the right document."""
    client, models = fake_client(['{"note": "- a ticket", "wrong_detail": "says Jira"}'])

    note = write_note(client, UsageMeter(), QUESTION, "DOCUMENT TEXT", wrong=True)

    assert note.wrong_detail == "says Jira"
    assert "DOCUMENT TEXT" in models.requests[0]
    assert QUESTION.text in models.requests[0]


def test_has_number() -> None:
    """Any digit in a note is flagged for review."""
    assert has_number("- the pass rate was 0.92")
    assert has_number("- ticket INC-482")
    assert not has_number("- a PR from the inference team, a few weeks back")


def test_write_each_keeps_going_and_saves_each() -> None:
    """A failing note is reported; the others are written and saved one by one."""
    saved: list[str] = []

    def write(question_id: str) -> MemoryNote:
        if question_id == "b":
            raise ValueError("no wrong detail")
        return MemoryNote(question_id=question_id, note="n")

    failed = write_each(["a", "b", "c"], write, lambda note: saved.append(note.question_id))

    assert saved == ["a", "c"]
    assert list(failed) == ["b"]
    assert "no wrong detail" in failed["b"]


def test_note_prompt_limits_depth() -> None:
    """The writer is told: two points, no deeper than the question, no solution."""
    text = prompt("memory_note_instructions").lower()

    assert "2 short bullet points" in text
    assert "no deeper than the question" in text
    assert "how it was solved" in text
