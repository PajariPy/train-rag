"""Tests for the Linear parser on real tickets."""

from collections.abc import Callable
from datetime import datetime

from train_rag.models import Artifact
from train_rag.parsers import LinearParser

LoadDocument = Callable[[str], Artifact]


def test_parses_description_and_dated_comments(real_document: LoadDocument) -> None:
    """`YYYY-MM-DD - Name:` comments give participants and the first activity date.

    Args:
        real_document: Fixture loader.
    """
    ticket = LinearParser().parse(real_document("linear_dated_comments"))

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
    ticket = LinearParser().parse(real_document("linear_dated_comments"))

    assert "Goal" not in ticket.participants
    assert "Goal: reduce noisy pages" in ticket.comments[0].text


def test_reads_description_from_body_key(real_document: LoadDocument) -> None:
    """Tickets that use `body:` instead of `description:` still get a description.

    Args:
        real_document: Fixture loader.
    """
    ticket = LinearParser().parse(real_document("linear_body_key"))

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
    ticket = LinearParser().parse(real_document("linear_background"))

    assert ticket.description is not None
    assert ticket.description.startswith("Customers are integrating Redwood streaming")
    assert not ticket.flagged


def test_reads_unlabeled_description(real_document: LoadDocument) -> None:
    """A ticket written as prose under `Summary:` uses that prose as description.

    Args:
        real_document: Fixture loader.
    """
    ticket = LinearParser().parse(real_document("linear_unlabeled"))

    assert ticket.description is not None
    assert ticket.description.startswith("Summary:\nBuild an end-to-end automated")
    assert not ticket.flagged
