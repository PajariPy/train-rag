"""Source-agnostic representation of a retrievable piece of content."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    """Kind of source an artifact came from."""

    SLACK = "slack"
    DOCUMENT = "document"
    MEETING_TRANSCRIPT = "meeting_transcript"
    MEETING_CHAT = "meeting_chat"
    URL = "url"
    PULL_REQUEST = "pull_request"


class Artifact(BaseModel):
    """A single retrievable item, normalized across all connectors."""

    id: str
    product: str
    source_type: SourceType
    content: str
    created_at: datetime | None = None
    author_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        """Build a corpus-wide unique key.

        Source ids are only unique within a product: the same public URL can
        appear in several products.

        Returns:
            The product and the source id joined with a slash.
        """
        return f"{self.product}/{self.id}"
