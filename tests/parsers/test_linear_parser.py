"""Tests for the Linear parser on real tickets."""

from collections.abc import Callable
from datetime import datetime

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import LinearTicket, OpenText
from train_rag.parsers import LinearParser

LoadDocument = Callable[[str], Artifact]


def parse_structured(artifact: Artifact) -> LinearTicket:
    """Parse a document that is in one of the formats, not open text.

    Args:
        artifact: Raw document.

    Returns:
        The parsed ticket.
    """
    parsed = LinearParser().parse(artifact)
    assert isinstance(parsed, LinearTicket)

    return parsed


def test_parses_description_and_dated_comments(real_document: LoadDocument) -> None:
    """`YYYY-MM-DD - Name:` comments give participants and the first activity date.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("linear_dated_comments"))

    assert ticket.description is not None
    assert ticket.description.startswith("Build a production-ready SLO-aware")
    assert ticket.participants == ["Evelyn Park", "Rajat Singh", "UX Review"]
    assert ticket.first_activity_at == datetime(2026, 3, 1)
    assert not ticket.flagged


def test_labels_inside_comments_are_not_people(real_document: LoadDocument) -> None:
    """`Goal:` inside a comment stays comment text instead of becoming an author.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("linear_dated_comments"))

    assert "Goal" not in ticket.participants
    assert "Goal: reduce noisy pages" in ticket.comments[0].text


def test_reads_description_from_body_key(real_document: LoadDocument) -> None:
    """Tickets that use `body:` instead of `description:` still get a description.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("linear_body_key"))

    assert ticket.description is not None
    assert ticket.description.startswith("Goal: Build a repeatable, safe automation pipeline")
    assert ("Lina Rodriguez", "SRE") in [
        (comment.author, comment.role) for comment in ticket.comments
    ]
    assert "Blocking" not in ticket.participants


def test_reads_background_section(real_document: LoadDocument) -> None:
    """A `background:` section stands in for the description.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("linear_background"))

    assert ticket.description is not None
    assert ticket.description.startswith("Customers are integrating Redwood streaming")
    assert not ticket.flagged


def test_reads_unlabeled_description(real_document: LoadDocument) -> None:
    """A ticket written as prose under `Summary:` uses that prose as description.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("linear_unlabeled"))

    assert ticket.description is not None
    assert ticket.description.startswith("Summary:\nBuild an end-to-end automated")
    assert not ticket.flagged


def test_keeps_long_document_in_no_format_as_open_text(real_document: LoadDocument) -> None:
    """A long ticket with no description in any format is kept whole as open text.

    Args:
        real_document: Fixture loader.
    """
    parsed = LinearParser().parse(real_document("linear_open_text"))

    assert isinstance(parsed, OpenText)
    assert parsed.text.startswith("design_notes:")
    assert parsed.useful
    assert not parsed.flagged


def test_marks_short_ticket_not_useful() -> None:
    """A short ticket in no known format is open text that is not useful, and not flagged."""
    artifact = Artifact(
        id="lin_short", source_type=SourceType.LINEAR, title="t", content="status: open"
    )

    parsed = LinearParser().parse(artifact)

    assert isinstance(parsed, OpenText)
    assert not parsed.useful
    assert not parsed.flagged


def test_reads_other_summary_section(real_document: LoadDocument) -> None:
    """A ticket with only a `postmortem_summary:` section uses it as description.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("linear_other_summary"))

    assert ticket.formats["description"] == "other_summary_section"
    assert ticket.description is not None
    assert ticket.description.startswith("On 2026-02-08 a rollout")
    assert not ticket.flagged
