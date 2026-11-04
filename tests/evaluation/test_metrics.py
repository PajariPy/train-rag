"""Tests for the benchmark metrics."""

import pytest

from train_rag.evaluation import document_recall, invalid_extra_docs, summarize_run
from train_rag.models import QuestionType
from train_rag.models.evaluation import AnswerGrade, QuestionResult


@pytest.mark.parametrize(
    ("expected", "retrieved", "recall"),
    [
        (["a", "b"], ["a", "b", "c"], 1.0),
        (["a", "b"], ["b"], 0.5),
        (["a", "b"], [], 0.0),
        (["a", "a"], ["a"], 1.0),
        ([], ["a"], None),
    ],
)
def test_document_recall(expected: list[str], retrieved: list[str], recall: float | None) -> None:
    """Recall counts distinct expected documents; it is undefined without any.

    Args:
        expected: Ground-truth ids.
        retrieved: Returned ids.
        recall: Expected recall.
    """
    assert document_recall(expected, retrieved) == recall


def test_invalid_extra_docs_counts_distinct_irrelevant_documents() -> None:
    """Only returned documents outside the expected set count, once each."""
    assert invalid_extra_docs(["a"], ["a", "b", "b", "c"]) == 2
    assert invalid_extra_docs(["a", "b"], ["a"]) == 0


def test_completeness_is_share_of_covered_facts() -> None:
    """Completeness is the covered share; without facts it follows correctness."""
    assert AnswerGrade(reasoning="", correct=True, facts_covered=[True, False]).completeness == 0.5
    assert AnswerGrade(reasoning="", correct=True).completeness == 1.0
    assert AnswerGrade(reasoning="", correct=False).completeness == 0.0


def test_score_is_zero_for_a_wrong_answer() -> None:
    """A wrong answer scores 0 however complete it is."""
    result = QuestionResult(
        question_id="q", question_type=QuestionType.BASIC, correct=False, completeness=1.0
    )

    assert result.score == 0.0


def test_summary_counts_errors_as_zero_and_skips_recall_without_documents() -> None:
    """Errors lower the averages; recall averages only questions that expect documents."""
    results = [
        QuestionResult(
            question_id="q1",
            question_type=QuestionType.BASIC,
            correct=True,
            completeness=0.5,
            document_recall=1.0,
            extra_docs=2,
        ),
        QuestionResult(question_id="q2", question_type=QuestionType.BASIC, error="judge: boom"),
        QuestionResult(
            question_id="q3",
            question_type=QuestionType.INFO_NOT_FOUND,
            correct=True,
            completeness=1.0,
        ),
    ]

    summary = summarize_run(results)

    assert summary.overall.questions == 3
    assert summary.overall.graded == 2
    assert summary.overall.errors == 1
    assert summary.overall.overall_score == pytest.approx(1.5 / 3)
    assert summary.overall.correctness == pytest.approx(2 / 3)
    assert summary.overall.document_recall == 1.0
    assert summary.overall.invalid_extra_docs == pytest.approx(2 / 3)
    assert summary.by_type["info_not_found"].document_recall is None
    assert summary.by_type["basic"].overall_score == pytest.approx(0.25)
