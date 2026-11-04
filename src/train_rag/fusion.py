"""Combine rankings: keep each document's best chunk, fuse rankings by reciprocal rank."""

from collections.abc import Mapping, Sequence

from train_rag.models.retrieval import RetrievedDocument, SearchHit

# Rank offset of reciprocal rank fusion: 60 is the value of the original paper (Cormack
# et al., 2009); it keeps one ranking's top hit from outweighing agreement between rankings.
RRF_K = 60


def best_per_document(hits: Sequence[SearchHit], limit: int) -> list[SearchHit]:
    """Rank documents by their closest chunk.

    Args:
        hits: Chunks from any number of tables, in any order.
        limit: Number of documents.

    Returns:
        The closest chunk of each document, closest first.
    """
    best: dict[int, SearchHit] = {}
    for hit in sorted(hits, key=lambda hit: hit.distance):
        best.setdefault(hit.document_id, hit)

    return list(best.values())[:limit]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[SearchHit]], limit: int, k: int = RRF_K
) -> list[RetrievedDocument]:
    """Fuse rankings: a document scores 1 / (k + rank) in every ranking that found it.

    Args:
        rankings: Ranked hits, best first and one per document, keyed by ranking name.
            A document's chunk position comes from the first ranking that found it.
        limit: Number of documents.
        k: Rank offset.

    Returns:
        The documents with the highest summed score, best first.
    """
    found: dict[int, RetrievedDocument] = {}
    for name, hits in rankings.items():
        for rank, hit in enumerate(hits, start=1):
            document = found.setdefault(
                hit.document_id,
                RetrievedDocument(
                    doc_id=hit.doc_id,
                    document_id=hit.document_id,
                    source_type=hit.source_type,
                    score=0.0,
                    ranks={},
                    position=hit.position,
                ),
            )
            document.score += 1 / (k + rank)
            document.ranks[name] = rank
    fused = sorted(found.values(), key=lambda document: document.score, reverse=True)

    return fused[:limit]
