"""Tests for evaluating questions with fake systems and judges."""

from typing import ClassVar

from train_rag.evaluation import (
    AnswerJudge,
    NoAnswerSystem,
    OracleSystem,
    RagSystem,
    evaluate_question,
    evaluate_questions,
)
from train_rag.models import Question, QuestionType
from train_rag.models.evaluation import AnswerGrade, SystemAnswer

QUESTIONS = [
    Question(
        id="qst_1",
        question_type=QuestionType.BASIC,
        text="Who owns the billing service?",
        gold_answer="Priya Desai owns it.",
        answer_facts=["The answer must name Priya Desai."],
        expected_doc_ids=["dsid_a"],
    ),
    Question(
        id="qst_2",
        question_type=QuestionType.INFO_NOT_FOUND,
        text="What is the CEO's shoe size?",
        gold_answer="This is not in the documents.",
        answer_facts=["The answer must say the information is not available."],
    ),
]


class ExactMatchJudge(AnswerJudge):
    """Correct when the answer equals the gold answer; every fact covered then."""

    name: ClassVar[str] = "exact"

    def grade(self, question: Question, answer: str) -> AnswerGrade:
        """Grade by exact match.

        Args:
            question: The question.
            answer: The system's answer.

        Returns:
            The grade.
        """
        correct = answer == question.gold_answer

        return AnswerGrade(
            reasoning="exact match",
            correct=correct,
            facts_covered=[correct] * len(question.answer_facts),
        )


class FailingJudge(AnswerJudge):
    """A judge whose call always fails."""

    name: ClassVar[str] = "failing"

    def grade(self, question: Question, answer: str) -> AnswerGrade:
        """Fail.

        Args:
            question: The question.
            answer: The system's answer.

        Raises:
            ValueError: Always.
        """
        raise ValueError("no verdict")


class CrashingSystem(RagSystem):
    """A system that always fails."""

    name: ClassVar[str] = "crashing"

    def answer(self, question: str) -> SystemAnswer:
        """Fail.

        Args:
            question: The question text.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("index missing")


def test_oracle_scores_full_marks() -> None:
    """The oracle returns the gold answer and expected documents."""
    result = evaluate_question(OracleSystem(QUESTIONS), ExactMatchJudge(), QUESTIONS[0])

    assert result.score == 1.0
    assert result.document_recall == 1.0
    assert result.extra_docs == 0
    assert result.error is None


def test_system_only_sees_the_question_text() -> None:
    """The oracle cannot answer a question it was not given the key for."""
    result = OracleSystem([]).answer(QUESTIONS[0].text)

    assert result.retrieved_doc_ids == []


def test_no_answer_system_retrieves_nothing() -> None:
    """The lower-bound system has zero recall on answerable questions."""
    result = evaluate_question(NoAnswerSystem(), ExactMatchJudge(), QUESTIONS[0])

    assert result.document_recall == 0.0
    assert result.correct is False


def test_system_failure_is_recorded_not_raised() -> None:
    """A crashing system leaves an ungraded result with the error."""
    result = evaluate_question(CrashingSystem(), ExactMatchJudge(), QUESTIONS[0])

    assert result.error == "system: RuntimeError('index missing')"
    assert result.score is None


def test_judge_failure_keeps_retrieval_metrics() -> None:
    """A failing judge leaves the answer and retrieval metrics but no grade."""
    result = evaluate_question(OracleSystem(QUESTIONS), FailingJudge(), QUESTIONS[0])

    assert result.error == "judge: ValueError('no verdict')"
    assert result.answer == "Priya Desai owns it."
    assert result.document_recall == 1.0
    assert result.correct is None


def test_results_keep_question_order_and_report_progress() -> None:
    """Parallel evaluation returns results in question order and reports each step."""
    calls: list[tuple[int, int]] = []

    results = evaluate_questions(
        OracleSystem(QUESTIONS),
        ExactMatchJudge(),
        QUESTIONS,
        workers=2,
        progress=lambda done, total: calls.append((done, total)),
    )

    assert [result.question_id for result in results] == ["qst_1", "qst_2"]
    assert calls == [(1, 2), (2, 2)]
