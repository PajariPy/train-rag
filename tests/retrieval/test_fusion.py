"""Tests for keeping the best chunk per document and reciprocal rank fusion."""

import pytest

from train_rag.fusion import best_per_document, reciprocal_rank_fusion
from train_rag.models import SourceType
from train_rag.models.retrieval import SearchHit


def hit(document_id: int, distance: float = 0.0, position: int = 0) -> SearchHit:
    """Build a search hit of a Slack document.

    Args:
        document_id: Document row id; the dataset id is derived from it.
        distance: Cosine distance, or negated BM25 score.
        position: Chunk position.

    Returns:
        The hit.
    """
    return SearchHit(
        doc_id=f"doc-{document_id}",
        document_id=document_id,
        source_type=SourceType.SLACK,
        field="chunk",
        position=position,
        text="",
        distance=distance,
    )


def test_best_per_document_keeps_the_closest_chunk() -> None:
    """Each document appears once, by its closest chunk, closest first."""
    hits = [hit(1, 0.5, 0), hit(2, 0.2, 0), hit(1, 0.1, 3), hit(3, 0.9)]

    best = best_per_document(hits, limit=2)

    assert [(found.document_id, found.position) for found in best] == [(1, 3), (2, 0)]


def test_fusion_rewards_agreement_between_rankings() -> None:
    """A document both rankings found beats one ranking's top hit."""
    fused = reciprocal_rank_fusion(
        {"gte": [hit(1), hit(2)], "bm25": [hit(3), hit(2)]}, limit=3, k=60
    )

    assert [document.document_id for document in fused] == [2, 1, 3]
    assert fused[0].score == pytest.approx(2 / 62)
    assert fused[0].ranks == {"gte": 2, "bm25": 2}
    assert fused[1].ranks == {"gte": 1}


def test_fusion_takes_the_chunk_from_the_first_ranking() -> None:
    """The first ranking that found a document sets its chunk position."""
    fused = reciprocal_rank_fusion(
        {"gte": [hit(1, position=4)], "bm25": [hit(1, position=0)]}, limit=1
    )

    assert fused[0].position == 4


def test_fusion_cuts_at_the_limit() -> None:
    """Only `limit` documents are returned."""
    fused = reciprocal_rank_fusion({"bm25": [hit(1), hit(2), hit(3)]}, limit=2)

    assert [document.document_id for document in fused] == [1, 2]
