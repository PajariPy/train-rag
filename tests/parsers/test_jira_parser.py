"""Tests for the Jira parser on real tickets."""

from collections.abc import Callable
from datetime import datetime

from train_rag.models import Artifact
from train_rag.parsers import JiraParser

LoadDocument = Callable[[str], Artifact]


def test_parses_inline_description_labels(real_document: LoadDocument) -> None:
    """`Issue summary:`, `Impact:` and `Environment:` are split out of the description.

    Args:
        real_document: Fixture loader.
    """
    ticket = JiraParser().parse(real_document("jira_inline_labels"))

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
    ticket = JiraParser().parse(real_document("jira_inline_labels"))
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
    ticket = JiraParser().parse(real_document("jira_issue_summary_block"))

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
    ticket = JiraParser().parse(real_document("jira_summary_long"))

    assert ticket.issue_summary is not None
    assert ticket.issue_summary.startswith("Request to implement a single, auditable automation")
    assert ticket.description == ticket.issue_summary
    assert not ticket.flagged


def test_reads_unlabeled_ticket(real_document: LoadDocument) -> None:
    """A ticket with no section keys uses its prose and its `Issue Summary` heading.

    Args:
        real_document: Fixture loader.
    """
    ticket = JiraParser().parse(real_document("jira_unlabeled"))

    assert ticket.description is not None
    assert ticket.description.startswith("Issue Summary\n- Customer (Acme AI")
    assert ticket.issue_summary is not None
    assert ticket.issue_summary.startswith("- Customer (Acme AI, route experiments beta)")
    assert not ticket.flagged
