"""Lookup of the parser responsible for each source type."""

from collections.abc import Iterable

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import ParsedDocument
from train_rag.parsers.base import DocumentParser


class ParserRegistry:
    """Map each source type to one parser; registering again replaces the parser."""

    def __init__(self, parsers: Iterable[DocumentParser[ParsedDocument]] = ()) -> None:
        """Register the initial parsers.

        Args:
            parsers: Parsers to register, at most one per source type.
        """
        self.parsers: dict[SourceType, DocumentParser[ParsedDocument]] = {}
        for parser in parsers:
            self.register(parser)

    def register(self, parser: DocumentParser[ParsedDocument]) -> None:
        """Register a parser, replacing any parser of the same source type.

        Args:
            parser: Parser to use for its source type.
        """
        self.parsers[parser.source_type] = parser

    def get(self, source_type: SourceType) -> DocumentParser[ParsedDocument] | None:
        """Find the parser of a source type.

        Args:
            source_type: Source to look up.

        Returns:
            The registered parser, or None when the source has no parser.
        """
        return self.parsers.get(source_type)

    def parse(self, artifact: Artifact) -> ParsedDocument | None:
        """Parse a document with the parser of its source type.

        Args:
            artifact: Raw document.

        Returns:
            The parsed document, or None when its source has no parser.
        """
        parser = self.get(artifact.source_type)
        if parser is None:
            return None

        return parser.parse(artifact)
