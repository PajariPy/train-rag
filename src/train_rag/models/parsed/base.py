"""Building blocks shared by every parsed document."""

from datetime import datetime

from pydantic import BaseModel, Field

from train_rag.models.artifact import SourceType


class Person(BaseModel):
    """A person as written in a document, with an optional role in parentheses."""

    name: str
    role: str | None = None


class Comment(BaseModel):
    """One comment from a ticket or pull request discussion."""

    author: str
    role: str | None = None
    posted_at: datetime | None = None
    text: str


class ParsedDocument(BaseModel):
    """Fields every parser returns, plus the expected fields it could not find.

    A document with missing fields is flagged for review instead of failing the run.
    `document_id` links back to the raw document (title and content) in the Postgres
    `documents` table; `doc_id` alone is not enough, as the dataset reuses a few ids.
    `formats` names, per field that comes in several formats, the parser method that
    found it, e.g. `{"channel": "channel_from_path"}`. `useful` is False for a
    document with too little content to index; it is not flagged, since there is
    nothing left to review.
    """

    doc_id: str
    source_type: SourceType
    document_id: int | None = None
    formats: dict[str, str] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    useful: bool = True

    @property
    def flagged(self) -> bool:
        """Tell whether the document needs review.

        Returns:
            True when at least one expected field is missing.
        """
        return bool(self.missing_fields)
