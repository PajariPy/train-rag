"""Tests for the Gmail parser on real threads in each stored format."""

from collections.abc import Callable

from train_rag.models import Artifact
from train_rag.parsers import GmailParser

LoadDocument = Callable[[str], Artifact]


def test_parses_python_list_thread(real_document: LoadDocument) -> None:
    """A thread stored as a Python list yields every email with its headers.

    Args:
        real_document: Fixture loader.
    """
    thread = GmailParser().parse(real_document("gmail_python_list"))
    first = thread.messages[0]

    assert len(thread.messages) == 4
    assert first.sender == "Camila Reyes <camila.reyes@redwoodinference.com>"
    assert first.recipients == ["Elena Morales <elena.morales@gmail.com>"]
    assert len(first.cc) == 3
    assert first.date == "Mon, 20 Dec 2026 09:12:00 -0800"
    assert thread.subject == "Candidate brief + reviewer slots for Senior Infra SWE - Morales"
    assert first.body.startswith("Hi Elena")
    assert not thread.flagged


def test_attachment_names_are_clean_file_names(real_document: LoadDocument) -> None:
    """Annotations like `(PDF)` and duplicates are dropped from attachment names.

    Args:
        real_document: Fixture loader.
    """
    thread = GmailParser().parse(real_document("gmail_python_list"))

    assert thread.attachments == ["infra_takehome.pdf", "scoring_sheet.xlsx"]


def test_parses_json_list_thread(real_document: LoadDocument) -> None:
    """A thread stored as valid JSON is split into its emails.

    Args:
        real_document: Fixture loader.
    """
    thread = GmailParser().parse(real_document("gmail_json_list"))

    assert len(thread.messages) == 4
    assert thread.subject == "Quick 15m guided onramp after your usage surge"
    assert thread.messages[1].sender == "Jamie Ross <jamie@brightmetrics.com>"
    assert not thread.flagged


def test_parses_mixed_quotes_list_thread(real_document: LoadDocument) -> None:
    """A list mixing `"..."` and `'...'` strings (not valid JSON) still yields every email.

    Args:
        real_document: Fixture loader.
    """
    thread = GmailParser().parse(real_document("gmail_mixed_quotes_list"))

    assert thread.subject == "Recap & next steps — reschedule technical deep dive"
    assert thread.participants[:3] == [
        "Kimberly Park <kimberly_park@redwoodinference.com>",
        "Tom Reyes <tom@novapulse.io>",
        "Ivy Chen <ivy@novapulse.io>",
    ]
    assert thread.attachments == ["deep_dive_agenda.pdf", "mutual_action_plan.xlsx"]
    assert not thread.flagged


def test_splits_plain_text_thread_on_from_lines(real_document: LoadDocument) -> None:
    """A plain-text thread is split into emails at each `From:` line.

    Args:
        real_document: Fixture loader.
    """
    thread = GmailParser().parse(real_document("gmail_plain_text"))

    assert [message.sender for message in thread.messages[:3]] == [
        "Vivek Kulkarni <vivek@redwood.com>",
        "Asha Patel <asha@finwave.ai>",
        "Marissa Cole <marissa@redwood.com>",
    ]
    assert thread.messages[1].subject == "Re: Q3 consumption audit + stair-step expansion proposal"
    assert "SLA_stairstep_amendment_draft.pdf" in thread.attachments
    assert not thread.flagged


def test_splits_numbered_body_sections(real_document: LoadDocument) -> None:
    """`message_N_body:` sections become emails; their missing headers stay flagged.

    Args:
        real_document: Fixture loader.
    """
    thread = GmailParser().parse(real_document("gmail_numbered_sections"))

    assert len(thread.messages) == 4
    assert thread.messages[0].body.startswith("Hi all,")
    assert thread.missing_fields == ["sender", "recipients", "subject"]
