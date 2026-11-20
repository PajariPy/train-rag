"""Tests for the union rerank, the BM25-only keyword query and the cached rerank scores."""

import asyncio
from collections.abc import Sequence

from train_rag.models import SourceType
from train_rag.models.retrieval import RetrievedDocument, SearchHit
from train_rag.pipelines.retrieval import CANDIDATES, RERANK_DEPTH, RetrievalPipeline
from train_rag.reranker import Reranker


class CountingReranker(Reranker):
    """Scores a text by its length and records every text it scored."""

    name = "counting"

    def __init__(self) -> None:
        """Start with nothing scored."""
        self.scored: list[str] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score each text by its length.

        Args:
            query: Ignored.
            texts: Document texts.

        Returns:
            One score per text.
        """
        self.scored.extend(texts)

        return [float(len(text)) for text in texts]


def hits(prefix: str, count: int, start: int) -> list[SearchHit]:
    """Build a ranking of Slack documents.

    Args:
        prefix: Start of each doc id.
        count: Number of hits.
        start: Row id of the first document.

    Returns:
        The hits, best first.
    """
    return [
        SearchHit(
            doc_id=f"{prefix}{index}",
            document_id=start + index,
            source_type=SourceType.SLACK,
            field="chunk",
            text="",
            distance=float(index),
        )
        for index in range(count)
    ]


class FakePipeline(RetrievalPipeline):
    """The pipeline with searches and database reads replaced by fixed results."""

    def __init__(self, union: bool) -> None:
        """Set up two disjoint rankings of `CANDIDATES` documents each.

        Args:
            union: Rerank the whole union.
        """
        self.reranker = CountingReranker()
        self.union = union
        self.model_lock = asyncio.Lock()
        self.keyword_queries: list[str] = []

    async def semantic_search(self, query: str) -> list[SearchHit]:
        """Return the fixed gte ranking."""
        return hits("gte", CANDIDATES, 0)

    async def keyword_search(self, query: str) -> list[SearchHit]:
        """Record the BM25 query and return the fixed BM25 ranking."""
        self.keyword_queries.append(query)

        return hits("bm25", CANDIDATES, 1000)

    async def describe(self, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        """Give every document a title and a text named after it."""
        return [
            document.model_copy(update={"title": document.doc_id, "text": "x"})
            for document in documents
        ]


def test_union_reranks_every_document_of_both_searches() -> None:
    """With union, the reranker reads both searches' documents, not only the fused 50."""
    pipeline = FakePipeline(union=True)
    asyncio.run(pipeline.run("question", 5))
    assert isinstance(pipeline.reranker, CountingReranker)
    assert len(pipeline.reranker.scored) == 2 * CANDIDATES


def test_fused_reranks_the_top_of_the_fusion() -> None:
    """Without union, the reranker reads the fused top `RERANK_DEPTH`."""
    pipeline = FakePipeline(union=False)
    asyncio.run(pipeline.run("question", 5))
    assert isinstance(pipeline.reranker, CountingReranker)
    assert len(pipeline.reranker.scored) == RERANK_DEPTH


def test_keyword_query_goes_to_bm25_only() -> None:
    """BM25 searches the keyword query; without one it searches the question."""
    pipeline = FakePipeline(union=True)
    asyncio.run(pipeline.run("the question", 5, keyword_query="the question plus rewrites"))
    asyncio.run(pipeline.run("the question", 5))
    assert pipeline.keyword_queries == ["the question plus rewrites", "the question"]


def test_rerank_cached_scores_each_document_once() -> None:
    """A document already scored for the query is not read by the reranker again."""
    pipeline = FakePipeline(union=True)
    documents = [
        RetrievedDocument(
            doc_id=doc_id, document_id=index, source_type=SourceType.SLACK, score=0.0, ranks={}
        )
        for index, doc_id in enumerate(["short", "a-longer-one"])
    ]
    scores: dict[str, float] = {}
    first = asyncio.run(pipeline.rerank_cached("q", documents[:1], scores))
    second = asyncio.run(pipeline.rerank_cached("q", documents, scores))
    assert isinstance(pipeline.reranker, CountingReranker)
    assert len(pipeline.reranker.scored) == 2
    assert [document.doc_id for document in first] == ["short"]
    assert [document.doc_id for document in second] == ["a-longer-one", "short"]
    assert set(scores) == {"short", "a-longer-one"}
