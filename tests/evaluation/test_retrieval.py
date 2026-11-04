"""Tests for the retrieval-only evaluation with a fake retriever."""

from typing import ClassVar

import pytest

from train_rag.evaluation.retrieval import evaluate_retrieval, format_retrieval_report
from train_rag.models import Question, QuestionType, SourceType
from train_rag.models.retrieval import SearchHit
from train_rag.retrieval import Retriever

QUESTIONS = [
    Question(
        id="qst_1",
        question_type=QuestionType.BASIC,
        text="Who owns billing?",
        gold_answer="Priya.",
        expected_doc_ids=["dsid_a", "dsid_b"],
    ),
    Question(
        id="qst_2",
        question_type=QuestionType.SEMANTIC,
        text="Why did the canary fail?",
        gold_answer="A stale digest.",
        expected_doc_ids=["dsid_c"],
    ),
    Question(
        id="qst_3",
        question_type=QuestionType.INFO_NOT_FOUND,
        text="What is the CEO's shoe size?",
        gold_answer="Not in the documents.",
    ),
]


class FixedRetriever(Retriever):
    """Returns the same ranked documents for every question."""

    name: ClassVar[str] = "fixed"

    def searchable_doc_ids(self) -> set[str]:
        """List the documents this retriever returns.

        Returns:
            The fixed doc ids.
        """
        return {"dsid_x", "dsid_a", "dsid_c"}

    def retrieve(self, question: str, limit: int) -> list[SearchHit]:
        """Return `dsid_x`, `dsid_a`, `dsid_c`, in that order.

        Args:
            question: Ignored.
            limit: Number of hits.

        Returns:
            The fixed hits, cut to `limit`.
        """
        return [
            SearchHit(
                doc_id=doc_id,
                document_id=index,
                source_type=SourceType.JIRA,
                field="issue_summary",
                text="",
                distance=index / 10,
            )
            for index, doc_id in enumerate(["dsid_x", "dsid_a", "dsid_c"])
        ][:limit]


def test_recall_at_k_counts_expected_documents_in_the_top_k() -> None:
    """Recall@k averages the share of expected documents found in the first k hits."""
    metrics = evaluate_retrieval(FixedRetriever(), QUESTIONS, {"dsid_a", "dsid_c"}, ks=(1, 2, 3))

    overall = metrics["all"]
    assert overall.questions == 2
    assert overall.recall_at == {1: 0.0, 2: pytest.approx(0.25), 3: pytest.approx(0.75)}
    assert metrics["basic"].recall_at[3] == pytest.approx(0.5)
    assert metrics["semantic"].recall_at[3] == 1.0


def test_reachable_is_the_share_of_expected_documents_in_the_index() -> None:
    """`dsid_b` is not indexed, so the basic question can reach at most half its documents."""
    metrics = evaluate_retrieval(FixedRetriever(), QUESTIONS, {"dsid_a", "dsid_c"}, ks=(3,))

    assert metrics["basic"].reachable == 0.5
    assert metrics["all"].reachable == pytest.approx(0.75)


def test_questions_without_expected_documents_are_left_out() -> None:
    """Info-not-found questions have no documents to find and get zero-question metrics."""
    metrics = evaluate_retrieval(FixedRetriever(), QUESTIONS, set(), ks=(1,))

    assert metrics["info_not_found"].questions == 0
    assert "| info_not_found | 0 | 0.0% | 0.0% |" in format_retrieval_report("t", metrics)
