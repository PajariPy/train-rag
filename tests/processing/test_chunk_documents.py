"""Tests for cutting cleaned documents into chunks."""

from train_rag.models.chunk import CleanDocument
from train_rag.processing.chunking import RecursiveChunker, chunk_documents


def test_chunks_keep_their_document_and_position() -> None:
    """Every chunk says which document it came from and where in it."""
    documents = [
        CleanDocument(document_id=1, doc_id="dsid_a", title="A", text="one two\n\nthree four"),
        CleanDocument(document_id=2, doc_id="dsid_b", title="B", text="five"),
    ]
    chunks = chunk_documents(RecursiveChunker(size=10, context=0), documents)

    assert [(chunk.doc_id, chunk.position, chunk.text) for chunk in chunks] == [
        ("dsid_a", 0, "A\none two"),
        ("dsid_a", 1, "A\nthree four"),
        ("dsid_b", 0, "B\nfive"),
    ]
