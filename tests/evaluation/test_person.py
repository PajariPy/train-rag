"""Tests for the simulated person who answers the agent from a memory note."""

from test_memory_notes import QUESTION, fake_client

from train_rag.evaluation.person import SimulatedPerson
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.memory import MemoryNote
from train_rag.prompts import prompt

NOTE = MemoryNote(question_id="qst_0178", note="- NOTE-POINT: it was a PR")


def test_person_replies_with_model_text() -> None:
    """The reply is the model's text."""
    client, _ = fake_client(["A PR, I think."])

    assert SimulatedPerson(client, UsageMeter(), QUESTION.text, NOTE).reply("PR or ticket?") == (
        "A PR, I think."
    )


def test_person_sees_note_and_history_never_gold() -> None:
    """Each call carries the note and the exchange so far, and never the gold answer."""
    client, models = fake_client(["A PR.", "No idea, sorry."])
    person = SimulatedPerson(client, UsageMeter(), QUESTION.text, NOTE)

    person.reply("Was it a PR or a ticket?")
    person.reply("Which team?")

    second = models.requests[1]
    assert "NOTE-POINT" in second
    assert "Was it a PR or a ticket?" in second and "A PR." in second and "Which team?" in second
    assert QUESTION.gold_answer not in second


def test_person_prompt_forbids_inventing() -> None:
    """The person is told to say they don't remember what the note doesn't cover."""
    text = prompt("person_instructions", question="q", note="n").lower()

    assert "don't remember" in text or "do not remember" in text
