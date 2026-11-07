"""Common interface for all document parsers."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import OpenText, ParsedDocument
from train_rag.parsers.text import MIN_USEFUL_CHARS, normalize_newlines


class DocumentParser[ParsedT: ParsedDocument](ABC):
    """Parse one source's raw documents into structured fields, without an LLM.

    Subclasses handle a single source type. A parser never raises on a malformed
    document: it returns what it could parse and lists the expected fields it could
    not find in `missing_fields`, so the document can be reviewed later.

    When a field is written in several formats, the parser has one method per format
    and tries them in order with `try_formats`; to support a new format, add a method
    and put it in that list. A document in none of the formats is returned as
    `OpenText` by `open_text`.
    """

    source_type: ClassVar[SourceType]

    @abstractmethod
    def parse(self, artifact: Artifact) -> ParsedT | OpenText:
        """Parse one document.

        Args:
            artifact: Raw document of this parser's source type.

        Returns:
            The parsed fields and the expected fields that are missing, or the whole
            document as open text when it is in none of the formats.
        """

    def open_text(self, artifact: Artifact) -> OpenText:
        """Keep a document in none of the formats as one block of open text.

        Args:
            artifact: Raw document.

        Returns:
            The whole text; not useful when under `MIN_USEFUL_CHARS` characters.
        """
        text = normalize_newlines(artifact.content).strip()

        return OpenText(
            doc_id=artifact.id,
            source_type=self.source_type,
            document_id=artifact.document_id,
            text=text,
            useful=len(text) >= MIN_USEFUL_CHARS,
        )

    @staticmethod
    def missing(**fields: object) -> list[str]:
        """List the names of expected fields that are empty.

        Args:
            **fields: Expected field values keyed by field name.

        Returns:
            Names of the fields whose value is None or empty.
        """
        return [name for name, value in fields.items() if value is None or value in ("", [])]

    @staticmethod
    def try_formats[InputT, ValueT](
        field: str,
        source: InputT,
        extractors: Sequence[Callable[[InputT], ValueT | None]],
        formats: dict[str, str],
    ) -> ValueT | None:
        """Try one method per format, in order, until one finds the field.

        Args:
            field: Name of the field being extracted.
            source: What the methods read: the title, the content or its sections.
            extractors: Format methods; each returns the value, or None or an empty
                value when the document is not in its format.
            formats: Format used per field; the winning method's name is recorded here.

        Returns:
            The first value found, or None when no format matches.
        """
        for extractor in extractors:
            value = extractor(source)
            if value:
                formats[field] = extractor.__name__

                return value

        return None
