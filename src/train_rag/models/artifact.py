"""Source-agnostic representation of a retrievable piece of content."""

from enum import StrEnum

from pydantic import BaseModel


class SourceType(StrEnum):
    """Kind of source an artifact came from."""

    SLACK = "slack"
    GMAIL = "gmail"
    LINEAR = "linear"
    GOOGLE_DRIVE = "google_drive"
    HUBSPOT = "hubspot"
    FIREFLIES = "fireflies"
    GITHUB = "github"
    JIRA = "jira"
    CONFLUENCE = "confluence"


class Artifact(BaseModel):
    """A single retrievable item, normalized across all data loaders.

    `document_id` is the row id in the Postgres `documents` table; it is set only
    when the artifact was loaded from the database.
    """

    id: str
    source_type: SourceType
    title: str
    content: str
    source_file: str | None = None
    row_index: int | None = None
    document_id: int | None = None
