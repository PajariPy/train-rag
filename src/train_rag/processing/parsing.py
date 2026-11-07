"""Parsing step: run each document's parser to get its typed fields."""

from collections.abc import Sequence

from train_rag.models.parsed import ParsedDocument
from train_rag.parsers import ParserRegistry
from train_rag.repository.document import to_artifact
from train_rag.tables import DocumentRecord


def parse_documents(
    registry: ParserRegistry, documents: Sequence[DocumentRecord]
) -> list[ParsedDocument]:
    """Parse raw documents with the parser of their source.

    Args:
        registry: Parsers by source type.
        documents: Raw documents.

    Returns:
        The parsed documents; sources without a parser (Google Drive, Confluence) give
        none.
    """
    return [
        parsed
        for document in documents
        if (parsed := registry.parse(to_artifact(document))) is not None
    ]
