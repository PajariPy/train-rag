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
    """A single retrievable item, normalized across all data loaders."""

    id: str
    source_type: SourceType
    title: str
    content: str
