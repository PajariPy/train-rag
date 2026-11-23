"""Tests for turning the documents a model cites into document ids."""

from train_rag.evaluation.pipeline_answer import (
    FineTunedPipelineAnswerSystem,
    PipelineAnswerSystem,
    cited_doc_ids,
    inline_doc_ids,
    link_citations,
)
from train_rag.evaluation.registry import SYSTEMS
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


def test_fine_tuned_system_is_registered_under_its_own_name() -> None:
    """The fine-tuned pipeline's runs don't mix with the base pipeline's."""
    assert FineTunedPipelineAnswerSystem.name == "hybrid-ft-rerank"
    assert FineTunedPipelineAnswerSystem.name != PipelineAnswerSystem.name
    assert FineTunedPipelineAnswerSystem.name in SYSTEMS


def test_inline_numbers_become_ids() -> None:
    """[2] becomes the second document's id; [1, 3] becomes two citations."""
    hits = [hit("dsid_a1"), hit("dsid_b2"), hit("dsid_c3")]

    answer = link_citations("Fixed by a retry [2]. Owners: Sam [1, 3].", hits)

    assert answer == "Fixed by a retry [dsid_b2]. Owners: Sam [dsid_a1][dsid_c3]."
    assert inline_doc_ids(answer) == ["dsid_b2", "dsid_a1", "dsid_c3"]


def test_out_of_range_inline_citations_are_removed() -> None:
    """A number the prompt never showed gives no citation; other brackets stay."""
    hits = [hit("dsid_a1")]

    assert link_citations("Done [7]. See [1, 9]. Array [x].", hits) == (
        "Done . See [dsid_a1]. Array [x]."
    )
