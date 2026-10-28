"""Tests for the Fireflies parser on real meetings in each header layout."""

from collections.abc import Callable
from datetime import date

import pytest

from train_rag.models import Artifact
from train_rag.models.parsed import Person
from train_rag.parsers import FirefliesParser

LoadDocument = Callable[[str], Artifact]


def test_parses_separate_header_fields(real_document: LoadDocument) -> None:
    """`Date:`, `Start:`, `Duration:` and `Attendees:` lines, summary and topics are read.

    Args:
        real_document: Fixture loader.
    """
    meeting = FirefliesParser().parse(real_document("fireflies_separate_fields"))

    assert meeting.held_on == date(2026, 2, 12)
    assert meeting.start_time == "15:00 PST"
    assert meeting.duration == "52 minutes"
    assert meeting.attendees[0] == Person(name="Ava Martinez", role="Redwood AE")
    assert len(meeting.attendees) == 7
    assert meeting.summary is not None
    assert meeting.summary.startswith("Quick scoping session to align on POC success criteria")
    assert meeting.topics[0] == "Success criteria and acceptance tests"
    assert len(meeting.topics) == 5
    assert not meeting.flagged


def test_parses_one_line_header_inside_transcript(real_document: LoadDocument) -> None:
    """A `Meeting header: Date: ...; Duration: ...; Attendees: ...` line is split up.

    Args:
        real_document: Fixture loader.
    """
    meeting = FirefliesParser().parse(real_document("fireflies_header_line"))

    assert meeting.held_on == date(2027, 3, 4)
    assert meeting.duration == "48 minutes"
    assert [person.name for person in meeting.attendees] == [
        "Maya",
        "Liam",
        "Sofia",
        "Rajiv",
        "Karen",
        "Ethan",
    ]
    assert not meeting.flagged


def test_parses_pipe_header_with_unlabeled_date(real_document: LoadDocument) -> None:
    """`Meeting header: 2026-04-18 15:00 PDT | Duration: 48m | Attendees: ...` is read.

    Args:
        real_document: Fixture loader.
    """
    meeting = FirefliesParser().parse(real_document("fireflies_pipe_header"))

    assert meeting.held_on == date(2026, 4, 18)
    assert meeting.duration == "48m"
    assert meeting.attendees[3] == Person(name="Mike Chen", role="Avalon CTO")
    assert not meeting.flagged


def test_parses_escaped_newlines(real_document: LoadDocument) -> None:
    """Content exported with literal backslash-n line breaks is normalized first.

    Args:
        real_document: Fixture loader.
    """
    meeting = FirefliesParser().parse(real_document("fireflies_escaped_newlines"))

    assert meeting.held_on == date(2028, 3, 16)
    assert meeting.start_time == "15:30 UTC"
    assert meeting.attendees[2] == Person(
        name="Dr. Serena Kaur", role="Aventra Head of Data Privacy"
    )
    assert meeting.topics[:2] == ["No-training guarantees", "data residency"]
    assert not meeting.flagged


def test_meeting_without_header_takes_attendees_from_transcript(
    real_document: LoadDocument,
) -> None:
    """With no header, attendees are the transcript speakers; the date stays flagged.

    Args:
        real_document: Fixture loader.
    """
    meeting = FirefliesParser().parse(real_document("fireflies_without_header"))

    assert meeting.summary is not None
    assert meeting.summary.startswith("Weekly POC sync to review load-test results")
    assert meeting.topics[0] == "Load test summary and spot checks"
    assert [person.name for person in meeting.attendees] == [
        "Jordan Patel",
        "Ellen Ruiz",
        "Tom Becker",
        "Maya Chen",
        "Priya Sharma",
        "Luis Ortega",
    ]
    assert meeting.missing_fields == ["held_on"]


def test_reads_date_from_header_label(real_document: LoadDocument) -> None:
    """A `Header: 2026-11-04 15:30 UTC | ...` line gives the meeting date.

    Args:
        real_document: Fixture loader.
    """
    meeting = FirefliesParser().parse(real_document("fireflies_header_label"))

    assert meeting.held_on == date(2026, 11, 4)
    assert len(meeting.attendees) == 7
    assert not meeting.flagged


def test_reads_unlabeled_attendee_groups(real_document: LoadDocument) -> None:
    """An `Attendees` line without a colon, then one line per company, gives attendees.

    Args:
        real_document: Fixture loader.
    """
    meeting = FirefliesParser().parse(real_document("fireflies_unlabeled_attendees"))

    assert [person.name for person in meeting.attendees][:3] == [
        "Liam O'Rourke",
        "Markus Klein",
        "Amina Yusuf",
    ]
    assert meeting.attendees[4].role == "Director, SecOps"
    assert not meeting.flagged


def test_parses_meeting_stored_as_python_list(real_document: LoadDocument) -> None:
    """A meeting stored as a list of strings is joined back into text first.

    Args:
        real_document: Fixture loader.
    """
    meeting = FirefliesParser().parse(real_document("fireflies_python_list"))

    assert meeting.held_on == date(2027, 11, 21)
    assert meeting.summary is not None
    assert meeting.summary.startswith("QBR-style health check")
    assert [person.name for person in meeting.attendees][:2] == ["Maya Chen", "Liam Ortega"]
    assert not meeting.flagged


@pytest.mark.parametrize(
    ("name", "held_on"),
    [
        ("fireflies_header_without_colon", date(2025, 3, 3)),
        ("fireflies_recorded_field", date(2025, 2, 7)),
        ("fireflies_start_field", date(2026, 9, 28)),
    ],
)
def test_reads_other_date_labels(real_document: LoadDocument, name: str, held_on: date) -> None:
    """`[Meeting header] 2025-...`, `Recorded:` and `Meeting start:` give the meeting date.

    Args:
        real_document: Fixture loader.
        name: Fixture file stem.
        held_on: Expected meeting date.
    """
    meeting = FirefliesParser().parse(real_document(name))

    assert meeting.held_on == held_on
    assert not meeting.flagged
