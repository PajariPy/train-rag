"""Parser for Gmail threads."""

import ast
import json
import re
from typing import ClassVar

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import EmailMessage, EmailThread
from train_rag.parsers.base import DocumentParser
from train_rag.parsers.text import normalize_newlines, split_sections, unique

HEADER = re.compile(r"^(?P<name>From|To|Cc|Date|Sent|Subject):\s*(?P<value>.*)$", re.IGNORECASE)
MESSAGE_START = re.compile(r"(?im)^(?=From:)")
NUMBERED_BODY_KEY = re.compile(r"^(?:messages?|msg|m)_?\d+_body$")
ATTACHMENTS = re.compile(r"(?im)^\s*Attach(?:ments?|ed)[^:\n]{0,30}:[ \t]*(?P<inline>.*)$")
BULLET = re.compile(r"^\s*[-*•]\s*(?P<item>.+)$")
FILE_NAME = re.compile(
    r"[\w\-.]+\.(?:pdf|docx?|xlsx?|csv|pptx?|zip|tar|gz|png|jpe?g|gif|txt|md|json|ya?ml|log)\b",
    re.IGNORECASE,
)


class GmailParser(DocumentParser[EmailThread]):
    """Parse a Gmail thread stored as a list of emails, or as plain text.

    Most threads are a Python or JSON list of email strings; the rest are plain text
    where each email starts at a `From:` line. Thread-level fields are flagged only
    when no email in the thread has them, since replies often drop headers.
    """

    source_type: ClassVar[SourceType] = SourceType.GMAIL

    def parse(self, artifact: Artifact) -> EmailThread:
        """Parse one Gmail thread.

        Args:
            artifact: Raw Gmail document.

        Returns:
            The thread with its emails and thread-level metadata.
        """
        formats: dict[str, str] = {}
        raw_messages = self.try_formats(
            "messages",
            artifact.content.strip(),
            [
                self.messages_from_json_list,
                self.messages_from_python_list,
                self.messages_from_numbered_sections,
                self.messages_from_text,
            ],
            formats,
        )
        messages = [self.message(raw) for raw in raw_messages or []]
        subject = next((message.subject for message in messages if message.subject), None)
        participants = unique(
            [
                person
                for message in messages
                for person in [message.sender or "", *message.recipients, *message.cc]
                if person
            ]
        )
        senders = [message.sender for message in messages if message.sender]
        recipients = [person for message in messages for person in message.recipients]

        return EmailThread(
            doc_id=artifact.id,
            source_type=self.source_type,
            document_id=artifact.document_id,
            formats=formats,
            subject=subject,
            participants=participants,
            attachments=unique([name for message in messages for name in message.attachments]),
            messages=messages,
            missing_fields=self.missing(
                messages=messages, sender=senders, recipients=recipients, subject=subject
            ),
        )

    @staticmethod
    def messages_from_json_list(content: str) -> list[str] | None:
        """Format 1: the thread is a JSON list of email strings.

        Args:
            content: Stripped Gmail document content.

        Returns:
            One string per email, or None when the content is not a JSON list.
        """
        try:
            decoded = json.loads(content)
        except ValueError:
            return None

        return string_list(decoded)

    @staticmethod
    def messages_from_python_list(content: str) -> list[str] | None:
        """Format 2: the thread is a Python list literal of email strings.

        Args:
            content: Stripped Gmail document content.

        Returns:
            One string per email, or None when the content is not a Python list.
        """
        if not content.startswith("["):
            return None

        try:
            decoded = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return None

        return string_list(decoded)

    @staticmethod
    def messages_from_numbered_sections(content: str) -> list[str] | None:
        """Format 3: one `message_1_body:`, `message_2_body:`, ... section per email.

        Also `messages_N_body`, `msg_N_body` and `mN_body`. Most of these emails carry
        only their body, so sender, recipients and subject stay missing; a quoted email
        inside a body that starts at a `From:` line is split off with its headers.

        Args:
            content: Stripped Gmail document content.

        Returns:
            One string per email, or None when there are no numbered body sections.
        """
        sections = split_sections(normalize_newlines(content))
        bodies = [body for key, body in sections.items() if NUMBERED_BODY_KEY.match(key)]

        return [
            email.strip() for body in bodies for email in MESSAGE_START.split(body) if email.strip()
        ] or None

    @staticmethod
    def messages_from_text(content: str) -> list[str] | None:
        """Format 4: plain text where each email starts at a `From:` line.

        Args:
            content: Stripped Gmail document content.

        Returns:
            One string per email; the whole text when there is no `From:` line.
        """
        text = normalize_newlines(content)
        chunks = [chunk for chunk in MESSAGE_START.split(text) if chunk.startswith("From:")]

        return chunks or [text]

    @staticmethod
    def message(raw: str) -> EmailMessage:
        """Parse one email's leading header block, attachments and body.

        Args:
            raw: Text of one email.

        Returns:
            The email with whichever headers it has.
        """
        headers: dict[str, str] = {}
        lines = raw.strip().splitlines()
        body_start = 0
        for index, line in enumerate(lines):
            match = HEADER.match(line.strip())
            if match is None:
                body_start = index
                break

            headers.setdefault(match.group("name").lower(), match.group("value").strip())
            body_start = index + 1

        return EmailMessage(
            sender=headers.get("from") or None,
            recipients=split_addresses(headers.get("to", "")),
            cc=split_addresses(headers.get("cc", "")),
            date=headers.get("date") or headers.get("sent") or None,
            subject=headers.get("subject") or None,
            attachments=attachment_names(raw),
            body="\n".join(lines[body_start:]).strip(),
        )


def string_list(decoded: object) -> list[str] | None:
    """Accept a decoded thread only if it is a list of email strings.

    Args:
        decoded: Value decoded from the document content.

    Returns:
        The emails with newlines normalized, or None for any other value.
    """
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        return None

    return [normalize_newlines(item) for item in decoded]


def split_addresses(value: str) -> list[str]:
    """Split a header like `Ann <ann@x.com>, bob@y.com` into addresses.

    Args:
        value: Header value.

    Returns:
        The trimmed, non-empty entries.
    """
    return [address.strip() for address in value.split(",") if address.strip()]


def attachment_names(raw: str) -> list[str]:
    """Collect attachment file names from `Attachments:` lines and their bullets.

    Only tokens that look like file names are kept, so prose around them and
    annotations like `(PDF)` are dropped.

    Args:
        raw: Text of one email.

    Returns:
        The distinct attachment file names in order.
    """
    names: list[str] = []
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        match = ATTACHMENTS.match(line)
        if match is None:
            continue

        names += FILE_NAME.findall(match.group("inline"))
        for following in lines[index + 1 :]:
            bullet = BULLET.match(following)
            if bullet is None:
                break

            names += FILE_NAME.findall(bullet.group("item"))

    return unique(names)
