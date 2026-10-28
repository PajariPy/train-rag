"""Tests for the HERB connector."""

import json
from pathlib import Path

import pytest

from herb_rag.connectors import HerbConnector
from herb_rag.models import QuestionType, SourceType

PRODUCT = {
    "team": ["eid_1"],
    "customers": ["CUST-0001"],
    "slack": [
        {
            "Channel": {"name": "planning-demo", "channelID": "ch-1"},
            "Message": {
                "User": {
                    "userId": "eid_1",
                    "timestamp": "2026-03-12T08:24:00",
                    "text": "Draft PRD is up.",
                    "utterranceID": "msg-1",
                },
                "Reactions": [],
            },
            "ThreadReplies": [],
            "id": "msg-1",
        }
    ],
    "documents": [
        {
            "id": "demo_prd",
            "content": "PRD body",
            "date": "2026-03-13T10:00:00",
            "author": "eid_1",
            "type": "Product Requirements Document",
            "document_link": "https://internal/docs/demo_prd",
            "feedback": "- Add metrics",
        }
    ],
    "meeting_transcripts": [],
    "meeting_chats": [],
    "urls": [{"id": "url-1", "link": "https://example.com", "description": "Demo video."}],
    "prs": [],
    "answerable_questions": [
        {
            "question": "Who wrote the PRD?",
            "ground_truth": "eid_1",
            "citations": ["demo_prd"],
            "type": "person",
        }
    ],
    "unanswerable_questions": ["Who are the competitors?"],
}


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Write a one-product HERB dataset to a temporary directory.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The dataset root.
    """
    (tmp_path / "metadata").mkdir()
    (tmp_path / "products").mkdir()
    employee = {"employee_id": "eid_1", "name": "Ann", "role": "PM", "location": "NY", "org": "x"}
    customer = {"id": "CUST-0001", "name": "Bob", "role": "CTO", "company": "Acme"}
    (tmp_path / "metadata" / "employee.json").write_text(json.dumps({"eid_1": employee}))
    (tmp_path / "metadata" / "customers_data.json").write_text(json.dumps([customer]))
    (tmp_path / "products" / "Demo.json").write_text(json.dumps(PRODUCT))

    return tmp_path


def test_load_normalizes_artifacts(data_dir: Path) -> None:
    """Artifacts from every source are normalized with product and metadata.

    Args:
        data_dir: Fixture dataset root.
    """
    corpus = HerbConnector(data_dir).load()

    by_type = {artifact.source_type: artifact for artifact in corpus.artifacts}
    assert set(by_type) == {SourceType.SLACK, SourceType.DOCUMENT, SourceType.URL}
    assert by_type[SourceType.SLACK].metadata["channel"] == "planning-demo"
    assert by_type[SourceType.DOCUMENT].metadata["feedback"] == "- Add metrics"
    assert by_type[SourceType.URL].key == "Demo/url-1"


def test_load_normalizes_questions(data_dir: Path) -> None:
    """String answers become lists and unanswerable questions are flagged.

    Args:
        data_dir: Fixture dataset root.
    """
    corpus = HerbConnector(data_dir).load()

    answerable, unanswerable = corpus.questions
    assert answerable.question_type is QuestionType.PERSON
    assert answerable.answers == ["eid_1"]
    assert answerable.citation_ids == ["demo_prd"]
    assert not unanswerable.answerable
    assert len(corpus.employees) == 1
    assert len(corpus.customers) == 1


def test_unknown_product_raises(data_dir: Path) -> None:
    """Requesting a product that is not on disk fails loudly.

    Args:
        data_dir: Fixture dataset root.
    """
    with pytest.raises(FileNotFoundError, match="Missing"):
        HerbConnector(data_dir, products=["Missing"]).load()
