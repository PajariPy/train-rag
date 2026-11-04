"""What flows between the ingestion steps: cleaned documents and their chunks."""

from pydantic import BaseModel


class CleanDocument(BaseModel):
    """A raw document after normalization, still linked to its row in `documents`."""

    document_id: int
    doc_id: str
    title: str
    text: str


class Chunk(BaseModel):
    """One piece of a document to embed: `position` counts from 0 within the document."""

    document_id: int
    doc_id: str
    position: int
    text: str
