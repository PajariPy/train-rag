"""Tests for the Gemini judge with a stub client (no API calls)."""

from types import SimpleNamespace
from typing import cast

import pytest
from google import genai

from train_rag.evaluation import GeminiJudge
from train_rag.evaluation.judge import grading_prompt
from train_rag.models import Question, QuestionType
from train_rag.models.evaluation import AnswerGrade

QUESTION = Question(
    id="qst_1",
    question_type=QuestionType.BASIC,
    text="Who owns the billing service?",
    gold_answer="Priya Desai owns it.",
    answer_facts=["The answer must name Priya Desai.", "The answer must say she owns it."],
)


def stub_client(parsed: AnswerGrade | None) -> genai.Client:
    """Build a client whose `generate_content` returns a fixed parsed verdict.

    Args:
        parsed: What the response's `parsed` attribute holds.

    Returns:
        The stub, typed as a client.
    """
    response = SimpleNamespace(parsed=parsed)
    models = SimpleNamespace(generate_content=lambda **_kwargs: response)

    return cast(genai.Client, SimpleNamespace(models=models))


def test_returns_the_parsed_verdict() -> None:
    """A verdict with one flag per fact is returned as is."""
    verdict = AnswerGrade(reasoning="names Priya", correct=True, facts_covered=[True, False])

    grade = GeminiJudge(stub_client(verdict)).grade(QUESTION, "Priya.")

    assert grade == verdict
    assert grade.completeness == 0.5


def test_rejects_wrong_number_of_fact_verdicts() -> None:
    """A verdict that skips a fact is an error, not a silently wrong completeness."""
    verdict = AnswerGrade(reasoning="", correct=True, facts_covered=[True])

    with pytest.raises(ValueError, match="1 fact verdicts for 2 facts"):
        GeminiJudge(stub_client(verdict)).grade(QUESTION, "Priya.")


def test_rejects_missing_verdict() -> None:
    """No parsed verdict is an error."""
    with pytest.raises(ValueError, match="No verdict"):
        GeminiJudge(stub_client(None)).grade(QUESTION, "Priya.")


def test_prompt_numbers_the_facts() -> None:
    """The judge sees the reference, numbered facts and the answer."""
    prompt = grading_prompt(QUESTION, "Priya.")

    assert "1. The answer must name Priya Desai." in prompt
    assert "2. The answer must say she owns it." in prompt
    assert "<reference_answer>\nPriya Desai owns it.\n</reference_answer>" in prompt
