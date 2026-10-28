"""Parsed HubSpot CRM records."""

from train_rag.models.parsed.base import ParsedDocument


class HubspotRecord(ParsedDocument):
    """A CRM record; its title is the company or deal name."""

    company: str | None = None
