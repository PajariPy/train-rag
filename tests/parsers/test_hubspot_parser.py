"""Tests for the HubSpot parser on a real CRM record."""

from collections.abc import Callable

from train_rag.models import Artifact
from train_rag.parsers import HubspotParser

LoadDocument = Callable[[str], Artifact]


def test_company_is_the_title(real_document: LoadDocument) -> None:
    """The record's title is its company or deal name.

    Args:
        real_document: Fixture loader.
    """
    record = HubspotParser().parse(real_document("hubspot_record"))

    assert record.company == "Sunrise Remedy Collective"
    assert not record.flagged


def test_flags_record_without_title(real_document: LoadDocument) -> None:
    """A record with an empty title is flagged.

    Args:
        real_document: Fixture loader.
    """
    document = real_document("hubspot_record").model_copy(update={"title": "  "})

    record = HubspotParser().parse(document)

    assert record.missing_fields == ["company"]
