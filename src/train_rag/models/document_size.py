"""Sizes of raw documents, as stored characters and as printed lines."""

from pydantic import BaseModel

from train_rag.models.artifact import SourceType


class DocumentSize(BaseModel):
    """How long one document is.

    `lines` counts the document laid out on a page: every line break starts a new line
    and a line longer than the page width wraps onto more lines.
    """

    document_id: int
    source_type: SourceType
    chars: int
    lines: int
