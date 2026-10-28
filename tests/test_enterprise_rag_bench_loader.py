"""Tests for the EnterpriseRAG-Bench data loader."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.models import QuestionType, SourceType

DOCUMENTS = [
    {
        "doc_id": "dsid_slack",
        "source_type": "slack",
        "title": "eng-runtime",
        "content": "sasha: p95 is up after the driver rollout.",
    },
    {
        "doc_id": "dsid_jira",
        "source_type": "jira",
        "title": "High P95 tail latency",
        "content": "description:\nCustomer enabled kv_prefetch.",
    },
]

QUESTIONS = [
    {
        "question_id": "qst_0001",
        "question_type": "basic",
        "source_types": ["slack"],
        "question": "What caused the p95 increase?",
        "expected_doc_ids": ["dsid_slack"],
        "gold_answer": "The driver rollout.",
        "answer_facts": ["The p95 increase followed the driver rollout."],
    },
    {
        "question_id": "qst_0481",
        "question_type": "info_not_found",
        "source_types": [],
        "question": "Which accounts were on the allowlist?",
        "expected_doc_ids": [],
        "gold_answer": "Not answerable from the documents.",
        "answer_facts": [],
    },
]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Write a tiny EnterpriseRAG-Bench dataset to a temporary directory.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The dataset root.
    """
    for name, rows in (("documents", DOCUMENTS), ("questions", QUESTIONS)):
        folder = tmp_path / "data" / name
        folder.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), folder / "test.parquet")

    return tmp_path


def test_load_normalizes_artifacts(data_dir: Path) -> None:
    """Documents become artifacts with typed sources.

    Args:
        data_dir: Fixture dataset root.
    """
    corpus = EnterpriseRagBenchLoader(data_dir).load()

    by_type = {artifact.source_type: artifact for artifact in corpus.artifacts}
    assert set(by_type) == {SourceType.SLACK, SourceType.JIRA}
    assert by_type[SourceType.SLACK].id == "dsid_slack"
    assert by_type[SourceType.JIRA].title == "High P95 tail latency"


def test_load_filters_source_types(data_dir: Path) -> None:
    """Only the requested sources are loaded, while questions stay complete.

    Args:
        data_dir: Fixture dataset root.
    """
    corpus = EnterpriseRagBenchLoader(data_dir, source_types=[SourceType.JIRA]).load()

    assert [artifact.id for artifact in corpus.artifacts] == ["dsid_jira"]
    assert len(corpus.questions) == 2


def test_load_normalizes_questions(data_dir: Path) -> None:
    """Questions keep their category, evidence and answerability.

    Args:
        data_dir: Fixture dataset root.
    """
    corpus = EnterpriseRagBenchLoader(data_dir).load()

    answerable, unanswerable = corpus.questions
    assert answerable.question_type is QuestionType.BASIC
    assert answerable.source_types == [SourceType.SLACK]
    assert answerable.expected_doc_ids == ["dsid_slack"]
    assert answerable.answerable
    assert not unanswerable.answerable


def test_missing_dataset_raises(tmp_path: Path) -> None:
    """Loading from an empty directory fails loudly.

    Args:
        tmp_path: Pytest temporary directory.
    """
    with pytest.raises(FileNotFoundError, match="train-rag-download"):
        EnterpriseRagBenchLoader(tmp_path).load()
