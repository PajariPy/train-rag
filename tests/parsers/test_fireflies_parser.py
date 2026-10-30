"""Tests for the Fireflies parser on real meetings in each header layout."""

from collections.abc import Callable
from datetime import date

import pytest

from train_rag.models import Artifact
from train_rag.models.parsed import FirefliesMeeting, InferredDate, OpenText, Person
from train_rag.parsers import FirefliesParser

LoadDocument = Callable[[str], Artifact]


def parse_structured(artifact: Artifact, parser: FirefliesParser | None = None) -> FirefliesMeeting:
    """Parse a meeting that has a summary, not open text.

    Args:
        artifact: Raw document.
        parser: Parser to use; a parser without inferred data when None.

    Returns:
        The parsed meeting.
    """
    parsed = (parser or FirefliesParser()).parse(artifact)
    assert isinstance(parsed, FirefliesMeeting)

    return parsed


def test_parses_separate_header_fields(real_document: LoadDocument) -> None:
    """`Date:`, `Start:`, `Duration:` and `Attendees:` lines, summary and topics are read.

    Args:
        real_document: Fixture loader.
    """
    meeting = parse_structured(real_document("fireflies_separate_fields"))

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
    meeting = parse_structured(real_document("fireflies_header_line"))

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
    meeting = parse_structured(real_document("fireflies_pipe_header"))

    assert meeting.held_on == date(2026, 4, 18)
    assert meeting.duration == "48m"
    assert meeting.attendees[3] == Person(name="Mike Chen", role="Avalon CTO")
    assert not meeting.flagged


def test_parses_escaped_newlines(real_document: LoadDocument) -> None:
    """Content exported with literal backslash-n line breaks is normalized first.

    Args:
        real_document: Fixture loader.
    """
    meeting = parse_structured(real_document("fireflies_escaped_newlines"))

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
    meeting = parse_structured(real_document("fireflies_without_header"))

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
    assert not meeting.flagged


def test_reads_date_from_header_label(real_document: LoadDocument) -> None:
    """A `Header: 2026-11-04 15:30 UTC | ...` line gives the meeting date.

    Args:
        real_document: Fixture loader.
    """
    meeting = parse_structured(real_document("fireflies_header_label"))

    assert meeting.held_on == date(2026, 11, 4)
    assert len(meeting.attendees) == 7
    assert not meeting.flagged


def test_reads_unlabeled_attendee_groups(real_document: LoadDocument) -> None:
    """An `Attendees` line without a colon, then one line per company, gives attendees.

    Args:
        real_document: Fixture loader.
    """
    meeting = parse_structured(real_document("fireflies_unlabeled_attendees"))

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
    meeting = parse_structured(real_document("fireflies_python_list"))

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
    meeting = parse_structured(real_document(name))

    assert meeting.held_on == held_on
    assert not meeting.flagged


INFERRED = InferredDate(
    held_on=date(2026, 3, 2),
    precision="week",
    evidence="send the pricing sheet by 2026-03-06",
    year_in_evidence=True,
)


def test_uses_inferred_date_when_header_has_none(real_document: LoadDocument) -> None:
    """A meeting without a header date takes the judge's date and is not flagged for it.

    Args:
        real_document: Fixture loader.
    """
    document = real_document("fireflies_without_header").model_copy(update={"document_id": 7})

    meeting = parse_structured(document, FirefliesParser({7: INFERRED}))

    assert meeting.held_on is None
    assert meeting.held_on_inferred == INFERRED
    assert "held_on" not in meeting.missing_fields


def test_header_date_wins_over_inferred_date(real_document: LoadDocument) -> None:
    """A header date is kept and an inferred date for the same meeting is ignored.

    Args:
        real_document: Fixture loader.
    """
    document = real_document("fireflies_header_label").model_copy(update={"document_id": 7})

    meeting = parse_structured(document, FirefliesParser({7: INFERRED}))

    assert meeting.held_on == date(2026, 11, 4)
    assert meeting.held_on_inferred is None


def test_meeting_without_any_date_is_not_flagged(real_document: LoadDocument) -> None:
    """Without a header date or an inferred one, the date stays empty and is not flagged.

    Args:
        real_document: Fixture loader.
    """
    meeting = parse_structured(real_document("fireflies_without_header"))

    assert meeting.held_on is None
    assert meeting.held_on_inferred is None
    assert "held_on" not in meeting.missing_fields


def test_uses_inferred_attendees_when_none_are_found(real_document: LoadDocument) -> None:
    """A meeting with no attendees takes the speakers an LLM listed and is not flagged.

    Args:
        real_document: Fixture loader.
    """
    content = "summary:\nQuick pricing sync.\n\ntranscript:\nwe agreed to send the deck."
    document = real_document("fireflies_without_header").model_copy(
        update={"document_id": 7, "content": content}
    )
    speakers = [Person(name="Maya Chen", role="AE")]

    meeting = parse_structured(document, FirefliesParser(inferred_attendees={7: speakers}))

    assert meeting.attendees == []
    assert meeting.attendees_inferred == speakers
    assert not meeting.flagged


def test_meeting_without_summary_is_open_text(real_document: LoadDocument) -> None:
    """A meeting with no Fireflies summary is kept as open text.

    Args:
        real_document: Fixture loader.
    """
    document = real_document("fireflies_without_header").model_copy(
        update={"content": "transcript:\n[00:01] Maya: hi all, quick sync on pricing."}
    )

    parsed = FirefliesParser().parse(document)

    assert isinstance(parsed, OpenText)
    assert not parsed.flagged


def test_summary_ends_before_transcript_lines(real_document: LoadDocument) -> None:
    """Without a `transcript:` key, the summary stops at the blank line before `[00:00]`.

    Args:
        real_document: Fixture loader.
    """
    meeting = parse_structured(real_document("fireflies_summary_before_transcript"))

    assert meeting.summary is not None
    assert meeting.summary.startswith("short technical deep-dive on private deployment")
    assert meeting.summary.endswith("runbook ops for key compromise.")


def test_topics_end_before_transcript_lines(real_document: LoadDocument) -> None:
    """Inline topics stop at the blank line before the transcript.

    Args:
        real_document: Fixture loader.
    """
    meeting = parse_structured(real_document("fireflies_topics_before_transcript"))

    assert meeting.topics == [
        "audit log schema",
        "admin activity trails",
        "export destinations",
        "retention and legal-hold",
        "encryption/KMS",
        "RBAC/SSO integration for admin auditing",
    ]


def test_header_field_stops_at_next_field_sentence(real_document: LoadDocument) -> None:
    """`Duration: ~46 minutes. Attendees: ...` gives only the duration.

    Args:
        real_document: Fixture loader.
    """
    meeting = parse_structured(real_document("fireflies_duration_run_on"))

    assert meeting.duration == "~46 minutes"


def test_keeps_summary_written_as_timed_agenda(real_document: LoadDocument) -> None:
    """A summary of `0:05 - Discussed ...` lines is kept: they are not transcript lines.

    Args:
        real_document: Fixture loader.
    """
    meeting = parse_structured(real_document("fireflies_agenda_summary"))

    assert meeting.summary is not None
    assert meeting.summary.startswith("0:00 - Quick intros and POC goals.")
    assert "0:05 - Discussed workload sizing" in meeting.summary
