"""Tests for the Jira parser on real tickets."""

from collections.abc import Callable
from datetime import datetime

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import JiraTicket, OpenText
from train_rag.parsers import JiraParser

LoadDocument = Callable[[str], Artifact]


def parse_structured(artifact: Artifact) -> JiraTicket:
    """Parse a document that is in one of the formats, not open text.

    Args:
        artifact: Raw document.

    Returns:
        The parsed ticket.
    """
    parsed = JiraParser().parse(artifact)
    assert isinstance(parsed, JiraTicket)

    return parsed


def test_parses_inline_description_labels(real_document: LoadDocument) -> None:
    """`Issue summary:`, `Impact:` and `Environment:` are split out of the description.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("jira_inline_labels"))

    assert ticket.issue_summary is not None
    assert ticket.issue_summary.startswith("After HealthNow applied the mixed-quant profile")
    assert ticket.issue_summary.endswith("latency increased by ~3-6x.")
    assert ticket.impact is not None
    assert ticket.impact.startswith("user-facing slowdowns for real-time document summarization")
    assert ticket.environment == (
        "dedicated cluster in us-east, serving model gpt-like-13b-quant-mix-v2."
    )
    assert not ticket.flagged


def test_role_first_comment_authors_are_swapped(real_document: LoadDocument) -> None:
    """`Support Engineer (Miguel Torres):` yields author Miguel Torres with that role.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("jira_inline_labels"))
    authors = [(comment.author, comment.role) for comment in ticket.comments]

    assert ("Miguel Torres", "Support Engineer") in authors
    assert ("Priya Menon", "SRE") in authors
    assert ticket.first_activity_at == datetime(2026, 3, 13, 9, 18)


def test_parses_issue_summary_block_and_parenthesized_authors(
    real_document: LoadDocument,
) -> None:
    """An `Issue Summary` line with bullets and `DATE (Name):` comments are understood.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("jira_issue_summary_block"))

    assert ticket.issue_summary is not None
    assert "On-Call has been paged repeatedly" in ticket.issue_summary
    assert ticket.participants == ["Aisha Bello", "Sean Gallagher", "Rafael Mendes"]
    assert ticket.first_activity_at == datetime(2026, 2, 12)
    assert not ticket.flagged


def test_reads_summary_long_section(real_document: LoadDocument) -> None:
    """A `summary_long:` section is both description and issue summary.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("jira_summary_long"))

    assert ticket.issue_summary is not None
    assert ticket.issue_summary.startswith("Request to implement a single, auditable automation")
    assert ticket.description == ticket.issue_summary
    assert not ticket.flagged


def test_reads_unlabeled_ticket(real_document: LoadDocument) -> None:
    """A ticket with no section keys uses its prose and its `Issue Summary` heading.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("jira_unlabeled"))

    assert ticket.description is not None
    assert ticket.description.startswith("Issue Summary\n- Customer (Acme AI")
    assert ticket.issue_summary is not None
    assert ticket.issue_summary.startswith("- Customer (Acme AI, route experiments beta)")
    assert not ticket.flagged


def test_keeps_long_document_in_no_format_as_open_text(real_document: LoadDocument) -> None:
    """A long ticket with no description in any format is kept whole as open text.

    Args:
        real_document: Fixture loader.
    """
    parsed = JiraParser().parse(real_document("jira_open_text"))

    assert isinstance(parsed, OpenText)
    assert parsed.text.startswith("requested_items:")
    assert parsed.useful
    assert not parsed.flagged


def test_marks_short_ticket_not_useful() -> None:
    """A short ticket in no known format is open text that is not useful, and not flagged."""
    artifact = Artifact(
        id="jira_short", source_type=SourceType.JIRA, title="t", content="status: open"
    )

    parsed = JiraParser().parse(artifact)

    assert isinstance(parsed, OpenText)
    assert not parsed.useful
    assert not parsed.flagged


def test_keeps_ticket_without_issue_summary_as_open_text(real_document: LoadDocument) -> None:
    """A ticket with a description but no issue summary in any format is open text.

    Args:
        real_document: Fixture loader.
    """
    parsed = JiraParser().parse(real_document("jira_without_issue_summary"))

    assert isinstance(parsed, OpenText)
    assert parsed.text.startswith("description:")
    assert parsed.useful
    assert not parsed.flagged


def test_reads_description_with_mixed_escaped_newlines(real_document: LoadDocument) -> None:
    """A description of escaped line breaks is normalized, even when the comments use real ones.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("jira_mixed_escapes"))

    assert ticket.issue_summary is not None
    assert ticket.issue_summary.startswith("Customers using LuminaSearch have reported")
    assert "Impact:" not in ticket.issue_summary
    assert ticket.impact is not None
    assert ticket.impact.startswith("- Production impact for LuminaSearch")


def test_heading_line_ends_inline_value(real_document: LoadDocument) -> None:
    """`Impact: ...` stops at a heading alone on its line, such as `Background/context:`.

    Args:
        real_document: Fixture loader.
    """
    ticket = parse_structured(real_document("jira_heading_after_impact"))

    assert ticket.impact is not None
    assert ticket.impact.endswith("prevent untagged seats that drive unbudgeted spend.")
