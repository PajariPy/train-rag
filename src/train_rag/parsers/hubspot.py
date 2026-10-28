"""Parser for HubSpot CRM records."""

from typing import ClassVar

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import HubspotRecord
from train_rag.parsers.base import DocumentParser


class HubspotParser(DocumentParser[HubspotRecord]):
    """Parse a CRM record; only the company or deal name (the title) for now."""

    source_type: ClassVar[SourceType] = SourceType.HUBSPOT

    def parse(self, artifact: Artifact) -> HubspotRecord:
        """Parse one CRM record.

        Args:
            artifact: Raw HubSpot document.

        Returns:
            The record with its company or deal name.
        """
        company = artifact.title.strip() or None

        return HubspotRecord(
            doc_id=artifact.id,
            source_type=self.source_type,
            document_id=artifact.document_id,
            company=company,
            missing_fields=self.missing(company=company),
        )
