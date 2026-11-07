"""Retrievers: find the documents most related to a question."""

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import bm25s
import Stemmer
from sqlalchemy import Engine
from sqlmodel import Session

from train_rag.embedder import Embedder
from train_rag.fusion import best_per_document
from train_rag.index_bm25 import DOCUMENTS_FILE, tokenize
from train_rag.models.retrieval import IndexedDocument, SearchHit
from train_rag.repository.embedding import embedded_doc_ids, embedded_sources, search_embeddings
from train_rag.tables import EMBEDDING_TABLES

CHUNKS_PER_DOCUMENT = 5


class Retriever(ABC):
    """Finds the documents most related to a question; implementations are interchangeable."""

    name: ClassVar[str]

    @abstractmethod
    def retrieve(self, question: str, limit: int) -> list[SearchHit]:
        """Find related documents.

        Args:
            question: The question text.
            limit: Number of documents.

        Returns:
            One hit per document, best first.
        """

    @abstractmethod
    def searchable_doc_ids(self) -> set[str]:
        """List the documents this retriever can find at all.

        Returns:
            The dataset ids of the indexed documents.
        """


class EmbeddingRetriever(Retriever):
    """Searches every embedding table with vectors of one field from the embedder.

    Each source's table is searched, then the hits of all sources are merged by cosine
    distance. Distances are comparable across tables because every table holds vectors
    from the same model. A document is ranked by its closest text (its best chunk), so
    `CHUNKS_PER_DOCUMENT` times more texts are fetched than documents asked for, leaving
    room for several chunks of one document.
    """

    name: ClassVar[str] = "embedding"

    def __init__(self, engine: Engine, embedder: Embedder, field: str) -> None:
        """Find which sources have embeddings of the field from this embedder.

        Args:
            engine: Database engine.
            embedder: The model the stored vectors come from.
            field: Embedded field to search, e.g. `summary` or `chunk`.
        """
        self.engine = engine
        self.embedder = embedder
        self.field = field
        self.tables = EMBEDDING_TABLES[embedder.dimensions]
        self.lock = threading.Lock()
        with Session(engine) as session:
            self.sources = embedded_sources(session, embedder.dimensions, embedder.name, field)

    def searchable_doc_ids(self) -> set[str]:
        """List the documents with embeddings of this retriever's field and model.

        Returns:
            The dataset ids across the searched tables.
        """
        with Session(self.engine) as session:
            return embedded_doc_ids(
                session, self.embedder.dimensions, self.embedder.name, self.field
            )

    def retrieve(self, question: str, limit: int) -> list[SearchHit]:
        """Embed the question and find the closest documents across all sources.

        Args:
            question: The question text.
            limit: Number of documents.

        Returns:
            The best hit of each document, closest first.
        """
        with self.lock:  # one model shared by the evaluation's worker threads
            vector = self.embedder.embed([question])[0]
        with Session(self.engine) as session:
            hits = [
                hit
                for source_type in self.sources
                for hit in search_embeddings(
                    session,
                    self.tables[source_type],
                    source_type,
                    self.embedder.name,
                    self.field,
                    vector,
                    limit * CHUNKS_PER_DOCUMENT,
                )
            ]

        return best_per_document(hits, limit)


class Bm25Retriever(Retriever):
    """Ranks whole documents by BM25 keyword match with the question.

    Words are lowercased and stemmed and English stopwords are dropped, as when the
    index was built. Hits carry no text; documents without a single matching word are
    left out.
    """

    name: ClassVar[str] = "bm25"

    def __init__(self, index_dir: Path) -> None:
        """Load an index saved by `index_bm25`.

        Args:
            index_dir: Folder of the index.
        """
        self.model = bm25s.BM25.load(index_dir)
        self.documents = [
            IndexedDocument.model_validate(item)
            for item in json.loads((index_dir / DOCUMENTS_FILE).read_text())
        ]
        self.stemmer = Stemmer.Stemmer("english")
        self.lock = threading.Lock()

    def searchable_doc_ids(self) -> set[str]:
        """List the indexed documents.

        Returns:
            Their dataset ids.
        """
        return {document.doc_id for document in self.documents}

    def retrieve(self, question: str, limit: int) -> list[SearchHit]:
        """Find the documents whose words best match the question.

        Args:
            question: The question text.
            limit: Number of documents.

        Returns:
            The best documents, highest score first; `distance` is the negated score.
        """
        with self.lock:  # the stemmer is shared by the evaluation's worker threads
            tokens = tokenize([question], self.stemmer)
            indexes, scores = self.model.retrieve(
                tokens, k=min(limit, len(self.documents)), show_progress=False
            )

        return [
            SearchHit(
                doc_id=document.doc_id,
                document_id=document.document_id,
                source_type=document.source_type,
                field="document",
                text="",
                distance=-float(score),
            )
            for index, score in zip(indexes[0], scores[0], strict=True)
            if score > 0
            for document in [self.documents[index]]
        ]
