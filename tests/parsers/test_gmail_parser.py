"""Tests for the Gmail parser on real threads in each stored format."""

from collections.abc import Callable

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import EmailThread, OpenText
from train_rag.parsers import GmailParser

LoadDocument = Callable[[str], Artifact]


def parse_structured(artifact: Artifact) -> EmailThread:
    """Parse a document that is in one of the formats, not open text.

    Args:
        artifact: Raw document.

    Returns:
        The parsed thread.
    """
    parsed = GmailParser().parse(artifact)
    assert isinstance(parsed, EmailThread)

    return parsed


def test_parses_python_list_thread(real_document: LoadDocument) -> None:
    """A thread stored as a Python list yields every email with its headers.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("gmail_python_list"))
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
    thread = parse_structured(real_document("gmail_python_list"))

    assert thread.attachments == ["infra_takehome.pdf", "scoring_sheet.xlsx"]


def test_parses_json_list_thread(real_document: LoadDocument) -> None:
    """A thread stored as valid JSON is split into its emails.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("gmail_json_list"))

    assert len(thread.messages) == 4
    assert thread.subject == "Quick 15m guided onramp after your usage surge"
    assert thread.messages[1].sender == "Jamie Ross <jamie@brightmetrics.com>"
    assert not thread.flagged


def test_parses_mixed_quotes_list_thread(real_document: LoadDocument) -> None:
    """A list mixing `"..."` and `'...'` strings (not valid JSON) still yields every email.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("gmail_mixed_quotes_list"))

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
    thread = parse_structured(real_document("gmail_plain_text"))

    assert [message.sender for message in thread.messages[:3]] == [
        "Vivek Kulkarni <vivek@redwood.com>",
        "Asha Patel <asha@finwave.ai>",
        "Marissa Cole <marissa@redwood.com>",
    ]
    assert thread.messages[1].subject == "Re: Q3 consumption audit + stair-step expansion proposal"
    assert "SLA_stairstep_amendment_draft.pdf" in thread.attachments
    assert not thread.flagged


def test_splits_numbered_body_sections(real_document: LoadDocument) -> None:
    """`message_N_body:` sections become emails; missing headers are not flagged.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("gmail_numbered_sections"))

    assert len(thread.messages) == 4
    assert thread.messages[0].body.startswith("Hi all,")
    assert thread.subject is None
    assert not thread.flagged


def test_reads_subject_after_bcc_header(real_document: LoadDocument) -> None:
    """A `Bcc:` line does not end the header block, so `Subject:` below it is read.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("gmail_bcc_header"))

    assert thread.subject == "Pricing & commit tiers: marketplace vs direct billing impact"
    assert thread.messages[0].body.startswith("Hi Daniel —")
    assert not thread.flagged


def test_skips_message_separator_lines(real_document: LoadDocument) -> None:
    """A `-----Message 1-----` line before the headers does not hide them.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("gmail_message_separators"))

    assert thread.messages[0].sender == "Olga Petrov <olga@redwood.ai>"
    assert thread.subject == "SSQ intake + request for annotated dataflow and KMS evidence"
    assert not thread.flagged


def test_keeps_list_of_bare_bodies_as_open_text(real_document: LoadDocument) -> None:
    """A list of email bodies without any header is open text.

    Args:
        real_document: Fixture loader.
    """
    parsed = GmailParser().parse(real_document("gmail_bodies_without_headers"))

    assert isinstance(parsed, OpenText)
    assert parsed.useful
    assert not parsed.flagged


def test_keeps_short_thread_as_open_text() -> None:
    """A thread under the length limit is open text that is not useful."""
    artifact = Artifact(
        id="gm_short", source_type=SourceType.GMAIL, title="t", content="['From: Kar']"
    )

    parsed = GmailParser().parse(artifact)

    assert isinstance(parsed, OpenText)
    assert not parsed.useful
    assert not parsed.flagged


def test_splits_recipients_on_semicolons(real_document: LoadDocument) -> None:
    """`To:` and `Cc:` lists separated by `;` give one entry per address.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("gmail_semicolon_recipients"))

    assert thread.messages[1].cc[:2] == [
        "Avery Johnson <avery.johnson@redwoodinference.com>",
        "Priyom Das <priyom.das@redwoodinference.com>",
    ]
    assert all(";" not in person for person in thread.participants)
