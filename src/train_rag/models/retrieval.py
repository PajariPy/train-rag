"""Results of searching the embedding tables."""

from pydantic import BaseModel

from train_rag.models.artifact import SourceType


class SearchHit(BaseModel):
    """One embedded text found for a query, closest first by cosine distance."""

    doc_id: str
    document_id: int
    source_type: SourceType
    field: str
    position: int = 0
    text: str
    distance: float


class IndexedDocument(BaseModel):
    """A document in a keyword (BM25) index, in index order."""

    document_id: int
    doc_id: str
    source_type: SourceType


class RetrievedDocument(BaseModel):
    """A document found for a query by the retrieval pipeline, best first.

    `score` is the document's reciprocal rank fusion score; `ranks` holds its rank in each
    ranking that found it (1 is the top), keyed by ranking name. `position` and `text` are
    its best chunk, or its first chunk when only keyword search found it. `rerank_score`
    is the reranker's relevance score, None when the pipeline ran without a reranker.
    """

    doc_id: str
    document_id: int
    source_type: SourceType
    score: float
    ranks: dict[str, int]
    position: int = 0
    title: str = ""
    text: str = ""
    rerank_score: float | None = None
