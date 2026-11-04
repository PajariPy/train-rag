"""Run the async retrieval pipeline behind the evaluation's synchronous `Retriever`."""

import asyncio
import threading
from typing import ClassVar

from train_rag.models.retrieval import RetrievedDocument, SearchHit
from train_rag.pipelines.retrieval import RetrievalPipeline
from train_rag.retrieval import Retriever


class PipelineRetriever(Retriever):
    """Evaluates the retrieval pipeline itself: BM25 and gte fused with RRF, maybe reranked.

    The pipeline runs on its own event loop in a background thread, so its database
    connections stay on one loop however the evaluation calls it.
    """

    name: ClassVar[str] = "hybrid"

    def __init__(self, pipeline: RetrievalPipeline) -> None:
        """Start the pipeline's event loop.

        Args:
            pipeline: Loaded retrieval pipeline.
        """
        self.pipeline = pipeline
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def searchable_doc_ids(self) -> set[str]:
        """List the documents BM25 indexed, which is every document of the corpus.

        Returns:
            Their dataset ids.
        """
        return self.pipeline.bm25.searchable_doc_ids()

    def retrieve(self, question: str, limit: int) -> list[SearchHit]:
        """Run the pipeline for one question.

        Args:
            question: The question text.
            limit: Number of documents.

        Returns:
            The documents, best first; `distance` is the negated rerank score, or the
            negated fusion score without a reranker.
        """
        future = asyncio.run_coroutine_threadsafe(self.pipeline.run(question, limit), self.loop)

        return to_hits(future.result())

    def retrieve_keywords(self, query: str, limit: int) -> list[SearchHit]:
        """Search by keywords alone (BM25), without fusion or reranking.

        Args:
            query: The keywords.
            limit: Number of documents.

        Returns:
            The documents, best first, with their text.
        """
        future = asyncio.run_coroutine_threadsafe(
            self.pipeline.run_keywords(query, limit), self.loop
        )

        return to_hits(future.result())


def to_hits(documents: list[RetrievedDocument]) -> list[SearchHit]:
    """Turn the pipeline's documents into the evaluation's search hits.

    Args:
        documents: Documents from the pipeline, best first.

    Returns:
        One hit per document; `distance` is the negated rerank score, or the negated
        fusion score without a reranker.
    """
    return [
        SearchHit(
            doc_id=document.doc_id,
            document_id=document.document_id,
            source_type=document.source_type,
            field="fused",
            position=document.position,
            text=document.text,
            distance=-(document.score if document.rerank_score is None else document.rerank_score),
        )
        for document in documents
    ]
