"""Tests for turning the documents a model cites into document ids."""

from train_rag.evaluation.pipeline_answer import cited_doc_ids
from train_rag.models.artifact import SourceType
from train_rag.models.retrieval import SearchHit


def hit(doc_id: str) -> SearchHit:
    """A retrieved document with the given id."""
    return SearchHit(
        doc_id=doc_id,
        document_id=1,
        source_type=SourceType.SLACK,
        field="content",
        text="text",
        distance=0.1,
    )


def test_cited_numbers_become_ids_in_cited_order() -> None:
    """Number 1 is the first document; the order of citing is kept."""
    hits = [hit("dsid_a"), hit("dsid_b"), hit("dsid_c")]

    assert cited_doc_ids([3, 1], hits) == ["dsid_c", "dsid_a"]


def test_out_of_range_and_repeated_numbers_are_dropped() -> None:
    """A number the prompt never showed, or a repeat, gives no id."""
    hits = [hit("dsid_a"), hit("dsid_b")]

    assert cited_doc_ids([0, 2, 2, 5], hits) == ["dsid_b"]
