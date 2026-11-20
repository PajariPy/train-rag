"""Retrieval: find documents for a query with BM25 and gte, fuse them by rank, rerank them.

For one query:

1. Normalize the query like the indexed text.
2. Search both ways at once: BM25 over whole documents, in a worker thread, and gte over
   chunks, embedding the query in a worker thread and then querying every source's
   table concurrently.
3. Keep each document's best gte chunk.
4. Fuse the two rankings with reciprocal rank fusion.
5. Load the title and best chunk text of each fused document.
6. Rerank: a cross-encoder reads the query with each of the top `RERANK_DEPTH` fused
   documents (or, with `union`, every document either search found) and orders them by
   its score.

BM25 may search another text: the question with Gemini's rewrites (`query_rewrite.py`).

The pipeline is async, so several queries can run at once on one loaded model and index.
"""

import asyncio
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import Session
from sqlmodel.ext.asyncio.session import AsyncSession

from train_rag.database import get_async_engine, get_engine
from train_rag.embedder import Embedder, GteModernBertEmbedder
from train_rag.fusion import best_per_document, reciprocal_rank_fusion
from train_rag.index_bm25 import bm25_index_dir
from train_rag.models import SourceType
from train_rag.models.retrieval import RetrievedDocument, SearchHit
from train_rag.pipelines.data_ingestion import FIELD
from train_rag.pipelines.describe import describe_documents
from train_rag.processing.normalize import normalize_text
from train_rag.repository.embedding import embedded_sources
from train_rag.repository.search import nearest_chunks
from train_rag.reranker import FULL_TOKENS, GteModernBertReranker, Reranker
from train_rag.retrieval import CHUNKS_PER_DOCUMENT, Bm25Retriever
from train_rag.tables import tables_for

# Documents taken from each ranking into the fusion.
CANDIDATES = 50
# Fused documents the reranker reads; the final results are picked from these.
RERANK_DEPTH = 50


class RetrievalPipeline:
    """Hybrid search over the whole corpus: BM25 and gte-modernbert chunks, fused with RRF.

    With a reranker, the fused documents are reordered by a cross-encoder.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        embedder: Embedder,
        bm25: Bm25Retriever,
        sources: list[SourceType],
        reranker: Reranker | None = None,
        union: bool = False,
    ) -> None:
        """Keep the loaded model and index for every query.

        Args:
            engine: Async database engine.
            embedder: Model the stored chunk vectors come from.
            bm25: Loaded BM25 index.
            sources: Sources with chunks from the embedder.
            reranker: Cross-encoder that reorders the fused documents, or None to keep
                the fusion's order.
            union: Rerank every document either search found (about 97) instead of the
                fused top `RERANK_DEPTH`.
        """
        self.engine = engine
        self.embedder = embedder
        self.bm25 = bm25
        self.sources = sources
        self.reranker = reranker
        self.union = union
        self.tables = tables_for(embedder.dimensions, embedder.finetuned)
        self.model_lock = asyncio.Lock()  # the GPU models are shared by concurrent queries

    @classmethod
    def load(
        cls,
        index_dir: Path | None = None,
        rerank: bool = True,
        embedder: Embedder | None = None,
        union: bool = False,
        rerank_tokens: int = FULL_TOKENS,
    ) -> "RetrievalPipeline":
        """Load gte-modernbert, the BM25 index of the whole corpus and the reranker.

        Args:
            index_dir: Folder of the BM25 index; the whole corpus's by default.
            rerank: Load gte-reranker-modernbert and rerank the fused documents.
            embedder: The embedder to search with; the base gte-modernbert by default. A
                fine-tuned one searches its own tables.
            union: Rerank the union of both searches instead of the fused top 50.
            rerank_tokens: Tokens of each document the reranker reads.

        Returns:
            The ready pipeline.
        """
        embedder = embedder or GteModernBertEmbedder()
        with Session(get_engine()) as session:
            sources = embedded_sources(
                session, tables_for(embedder.dimensions, embedder.finetuned), embedder.name, FIELD
            )

        return cls(
            get_async_engine(),
            embedder,
            Bm25Retriever(index_dir or bm25_index_dir(None)),
            sources,
            GteModernBertReranker(max_tokens=rerank_tokens) if rerank else None,
            union,
        )

    async def run(
        self, query: str, limit: int = 10, keyword_query: str | None = None
    ) -> list[RetrievedDocument]:
        """Find the documents for a query.

        Args:
            query: The question or search text.
            limit: Number of documents.
            keyword_query: What BM25 searches instead of the query (the query with
                rewrites appended), or None for the query itself. gte and the reranker
                always read the query.

        Returns:
            The documents, best first, with their titles and best chunks.
        """
        # 1. Normalize
        clean = normalize_text(query)
        # 2. Search, 3. best chunk per document
        semantic, keyword = await asyncio.gather(
            self.semantic_search(clean),
            self.keyword_search(normalize_text(keyword_query) if keyword_query else clean),
        )
        # 4. Fuse; gte comes first so its best chunk describes the document. The union
        # keeps every document of both searches for the reranker.
        depth = RERANK_DEPTH if self.reranker else limit
        if self.reranker and self.union:
            depth = 2 * CANDIDATES
        fused = reciprocal_rank_fusion({"gte": semantic, "bm25": keyword}, max(depth, limit))
        # 5. Load
        documents = await self.describe(fused)
        # 6. Rerank
        if self.reranker is None:
            return documents

        return (await self.rerank(self.reranker, clean, documents))[:limit]

    async def run_keywords(self, query: str, limit: int = 10) -> list[RetrievedDocument]:
        """Find documents by keywords alone (BM25), for exact names, ids and rare terms.

        Args:
            query: The keywords.
            limit: Number of documents.

        Returns:
            The documents, best first, with their titles and first chunks.
        """
        hits = await self.keyword_search(normalize_text(query))

        return await self.describe(reciprocal_rank_fusion({"bm25": hits}, limit))

    async def keyword_search(self, query: str) -> list[SearchHit]:
        """Rank whole documents by BM25.

        Args:
            query: Normalized query.

        Returns:
            Up to `CANDIDATES` documents, best first.
        """
        return await asyncio.to_thread(self.bm25.retrieve, query, CANDIDATES)

    async def semantic_search(self, query: str) -> list[SearchHit]:
        """Rank documents by their chunk closest to the query, across every source.

        Args:
            query: Normalized query.

        Returns:
            The best chunk of up to `CANDIDATES` documents, closest first.
        """
        async with self.model_lock:
            vectors = await asyncio.to_thread(self.embedder.embed, [query])
        per_source = await asyncio.gather(
            *(self.search_source(source_type, vectors[0]) for source_type in self.sources)
        )

        return best_per_document([hit for hits in per_source for hit in hits], CANDIDATES)

    async def search_source(self, source_type: SourceType, vector: list[float]) -> list[SearchHit]:
        """Find the closest chunks in one source's table.

        Args:
            source_type: Source to search.
            vector: Query vector.

        Returns:
            Enough chunks for `CANDIDATES` documents, closest first.
        """
        async with AsyncSession(self.engine) as session:
            return await nearest_chunks(
                session,
                self.tables[source_type],
                source_type,
                self.embedder.name,
                FIELD,
                vector,
                CANDIDATES * CHUNKS_PER_DOCUMENT,
            )

    async def describe(self, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        """Add each document's title and the text of its chunk.

        Args:
            documents: Fused documents.

        Returns:
            The same documents, in order, with title and text.
        """
        return await describe_documents(
            self.engine, self.tables, self.embedder.name, FIELD, documents
        )

    async def rerank_cached(
        self, query: str, documents: list[RetrievedDocument], scores: dict[str, float]
    ) -> list[RetrievedDocument]:
        """Rerank documents, scoring only the ones not scored for this query before.

        Documents without text (found by another search than this pipeline's) are
        described first, from their chunk at `position`.

        Args:
            query: The question or search text, as asked.
            documents: Documents to order, with or without text.
            scores: Reranker scores of this query by doc id, from earlier calls; the new
                scores are added to it.

        Returns:
            The documents with their scores, most relevant first.

        Raises:
            ValueError: If the pipeline has no reranker.
        """
        if self.reranker is None:
            raise ValueError("The pipeline was loaded without a reranker")

        new = [document for document in documents if document.doc_id not in scores]
        unread = [document for document in new if not document.text and not document.title]
        described = {document.doc_id: document for document in await self.describe(unread)}
        new = [described.get(document.doc_id, document) for document in new]
        if new:
            reranked = await self.rerank(self.reranker, normalize_text(query), new)
            scores.update({document.doc_id: document.rerank_score or 0.0 for document in reranked})

        scored = [
            described.get(document.doc_id, document).model_copy(
                update={"rerank_score": scores[document.doc_id]}
            )
            for document in documents
        ]

        return sorted(scored, key=lambda document: document.rerank_score or 0.0, reverse=True)

    async def rerank(
        self, reranker: Reranker, query: str, documents: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        """Order documents by the reranker's score of their title and text.

        Args:
            reranker: Cross-encoder to score with.
            query: Normalized query.
            documents: Documents with title and text.

        Returns:
            The documents with their scores, most relevant first.
        """
        texts = [f"{document.title}\n{document.text}" for document in documents]
        async with self.model_lock:
            scores = await asyncio.to_thread(reranker.score, query, texts)
        scored = [
            document.model_copy(update={"rerank_score": score})
            for document, score in zip(documents, scores, strict=True)
        ]

        return sorted(scored, key=lambda document: document.rerank_score or 0.0, reverse=True)

    async def close(self) -> None:
        """Close the database connections."""
        await self.engine.dispose()

