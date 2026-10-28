"""Tests for swapping and looking up parsers."""

from collections.abc import Callable
from typing import ClassVar

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import HubspotRecord
from train_rag.parsers import DocumentParser, HubspotParser, default_registry

LoadDocument = Callable[[str], Artifact]


class UppercaseHubspotParser(DocumentParser[HubspotRecord]):
    """A stand-in replacement parser used to test swapping."""

    source_type: ClassVar[SourceType] = SourceType.HUBSPOT

    def parse(self, artifact: Artifact) -> HubspotRecord:
        """Return the title in upper case as the company.

        Args:
            artifact: Raw HubSpot document.

        Returns:
            The record with an upper-cased company name.
        """
        return HubspotRecord(
            doc_id=artifact.id, source_type=self.source_type, company=artifact.title.upper()
        )


def test_default_registry_routes_by_source_type(real_document: LoadDocument) -> None:
    """Each document goes to the parser of its source type.

    Args:
        real_document: Fixture loader.
    """
    registry = default_registry()

    parsed = registry.parse(real_document("hubspot_record"))

    assert isinstance(registry.get(SourceType.HUBSPOT), HubspotParser)
    assert isinstance(parsed, HubspotRecord)


def test_register_replaces_parser(real_document: LoadDocument) -> None:
    """Registering a parser for a source type replaces the default one.

    Args:
        real_document: Fixture loader.
    """
    registry = default_registry()
    registry.register(UppercaseHubspotParser())

    parsed = registry.parse(real_document("hubspot_record"))

    assert isinstance(parsed, HubspotRecord)
    assert parsed.company == "SUNRISE REMEDY COLLECTIVE"


def test_sources_without_parser_return_none(real_document: LoadDocument) -> None:
    """Google Drive and Confluence have no parser yet.

    Args:
        real_document: Fixture loader.
    """
    document = real_document("hubspot_record").model_copy(
        update={"source_type": SourceType.CONFLUENCE}
    )

    assert default_registry().parse(document) is None


def test_parsed_document_links_to_stored_document(real_document: LoadDocument) -> None:
    """Every parser copies the documents row id, so the raw content can be looked up.

    Args:
        real_document: Fixture loader.
    """
    registry = default_registry()
    names = [
        "slack_channel",
        "gmail_mixed_quotes_list",
        "linear_body_key",
        "jira_inline_labels",
        "fireflies_header_line",
        "github_review_comments",
        "hubspot_record",
    ]
    for name in names:
        document = real_document(name).model_copy(update={"document_id": 42})

        parsed = registry.parse(document)

        assert parsed is not None
        assert parsed.document_id == 42
