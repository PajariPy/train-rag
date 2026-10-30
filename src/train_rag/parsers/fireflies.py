"""Parser for Fireflies meeting transcripts."""

import ast
import re
from collections.abc import Mapping
from datetime import datetime
from typing import ClassVar

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import FirefliesMeeting, InferredDate, OpenText, Person
from train_rag.parsers.base import DocumentParser
from train_rag.parsers.people import split_people
from train_rag.parsers.text import normalize_newlines, parse_datetime

FIELD_PREFIX = r"(?:^|[;|\n]|header:)[ \t]*[-*]?[ \t]*"
MEETING_HEADER_LINE = re.compile(r"(?im)^[ \t]*meeting header[ \t]*:(?P<value>.*)$")
HEADER_LINE = re.compile(r"(?im)^[ \t]*header[ \t]*:(?P<value>.*)$")
HEADER_WITHOUT_COLON = re.compile(
    r"(?im)^[ \t]*\[?meeting header\]?[ \t]*[-–—]?[ \t]*(?P<value>\d.*)$"
)
UNLABELED_ATTENDEES = re.compile(r"(?im)^[ \t]*(?:attendees|participants)[ \t]*$")
GROUP_LABEL = re.compile(r"^[^():,]{1,40}:\s*")
TRANSCRIPT_SPEAKER = re.compile(
    r"^\s*[\[(]?\d{1,2}:\d{2}(?::\d{2})?[\])]?\s*[-–—]?\s*(?:\[[^\]\n]{1,30}\]\s*)?"
    r"(?P<name>[A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})(?:\s*\((?P<role>[^)\n]{1,60})\))?:"
)
PLACEHOLDER_SPEAKER = re.compile(
    r"(?i)^(?:(?:speaker|unknown|participant)\b.*|(?:system |transcript )?notes?|all|everyone"
    r"|host|moderator|teams?|group|\w*bot)$"
)
SUMMARY_LABEL = re.compile(
    r"(?im)^[ \t]*(?:[A-Za-z]+[ /-]){0,2}summary(?:[ \t]*\([^)\n]*\))?"
    r"[ \t]*(?::[ \t]*(?P<inline>.*))?$"
)
TOPICS_LABEL = re.compile(
    r"(?im)^[ \t]*topics(?: covered)?(?:[ \t]*\([^)\n]*\))?[ \t]*:[ \t]*(?P<inline>.*)$"
)
LABEL_LINE = re.compile(r"^\s*[A-Za-z][\w()/ &-]{0,40}:(?:\s|$)")
BRACKETED_TIMESTAMP = re.compile(r"^\s*\[\d{1,2}:\d{2}")
NEXT_FIELD = re.compile(r"\.\s+[A-Z][\w /()-]{0,30}:.*$")
BULLET = re.compile(r"^\s*[-*•]\s*")


class FirefliesParser(DocumentParser[FirefliesMeeting]):
    """Parse a meeting: the Fireflies summary, topics and header fields.

    The header appears either as one `Meeting header: Date: ...; Start: ...;
    Duration: ...; Attendees: ...` line or as separate `Date:`, `Attendees:` lines,
    and sometimes sits inside the `transcript:` section. Fields are matched as labels
    at the start of a line anywhere in the document; transcript lines start with a
    timestamp or speaker, so they do not collide.

    A meeting without a Fireflies summary is kept as open text. When the header has no
    date, the date an LLM judge estimated from the text is attached as
    `held_on_inferred`; a meeting with neither keeps no date and is not flagged, since
    there is nothing left to recover it from. When no attendee list or transcript
    speaker is found, the speakers an LLM listed are attached as `attendees_inferred`;
    only a meeting with neither is flagged.
    """

    source_type: ClassVar[SourceType] = SourceType.FIREFLIES

    def __init__(
        self,
        inferred_dates: Mapping[int, InferredDate] | None = None,
        inferred_attendees: Mapping[int, list[Person]] | None = None,
    ) -> None:
        """Set up the parser.

        Args:
            inferred_dates: Estimated meeting dates keyed by document id, from
                `load_inferred_dates`; None when they are not loaded.
            inferred_attendees: Speakers an LLM listed, keyed by document id, from
                `load_inferred_attendees`; None when they are not loaded.
        """
        self.inferred_dates = inferred_dates or {}
        self.inferred_attendees = inferred_attendees or {}

    def parse(self, artifact: Artifact) -> FirefliesMeeting | OpenText:
        """Parse one meeting.

        Args:
            artifact: Raw Fireflies document.

        Returns:
            The meeting with summary, topics, attendees, date, start time and duration,
            or the whole text as open text when there is no summary.
        """
        formats: dict[str, str] = {}
        header = (
            self.try_formats(
                "content",
                artifact.content,
                [self.content_from_python_list, self.content_from_text],
                formats,
            )
            or ""
        )
        summary = labeled_block(header, SUMMARY_LABEL)
        if summary is None:
            return self.open_text(artifact)

        held_at = self.try_formats(
            "held_on",
            header,
            [
                self.date_from_date_field,
                self.date_from_meeting_header,
                self.date_from_header_line,
                self.date_from_header_without_colon,
                self.date_from_recorded_field,
                self.date_from_start_field,
            ],
            formats,
        )
        attendees = (
            self.try_formats(
                "attendees",
                header,
                [
                    self.attendees_from_labels,
                    self.attendees_from_unlabeled_list,
                    self.attendees_from_transcript,
                ],
                formats,
            )
            or []
        )
        held_on = held_at.date() if held_at else None
        document_id = artifact.document_id
        inferred_date = (
            None if held_on or document_id is None else self.inferred_dates.get(document_id)
        )
        inferred_attendees = (
            [] if attendees or document_id is None else self.inferred_attendees.get(document_id, [])
        )

        return FirefliesMeeting(
            doc_id=artifact.id,
            source_type=self.source_type,
            document_id=artifact.document_id,
            formats=formats,
            summary=summary,
            topics=topics(header),
            attendees=attendees,
            attendees_inferred=inferred_attendees,
            held_on=held_on,
            held_on_inferred=inferred_date,
            start_time=header_field(header, "start time|start|time"),
            duration=header_field(header, "duration"),
            missing_fields=self.missing(attendees=attendees or inferred_attendees),
        )

    @staticmethod
    def date_from_date_field(header: str) -> datetime | None:
        """Format 1: a `Date:` or `Date/Time:` field, on its own line or in a `;`/`|` list.

        Args:
            header: Normalized meeting text.

        Returns:
            The meeting date and time, or None when there is no readable date field.
        """
        date_text = header_field(header, "date/time|date")

        return parse_datetime(date_text) if date_text else None

    @staticmethod
    def date_from_meeting_header(header: str) -> datetime | None:
        """Format 2: a `Meeting header:` line that starts with an unlabeled date.

        Args:
            header: Normalized meeting text.

        Returns:
            The meeting date and time, or None when there is no such line.
        """
        date_text = meeting_header_line(header)

        return parse_datetime(date_text) if date_text else None

    @staticmethod
    def date_from_header_line(header: str) -> datetime | None:
        """Date format 3: a `Header: 2026-11-04 15:30 UTC | Duration ...` line.

        Args:
            header: Normalized meeting text.

        Returns:
            The meeting date and time, or None when there is no such line.
        """
        match = HEADER_LINE.search(header)

        return parse_datetime(match.group("value")) if match else None

    @staticmethod
    def date_from_header_without_colon(header: str) -> datetime | None:
        """Date format 4: `[Meeting header] 2025-03-03 ...` or `Meeting header - 2027-05-04 ...`.

        Args:
            header: Normalized meeting text.

        Returns:
            The meeting date and time, or None when there is no such line.
        """
        match = HEADER_WITHOUT_COLON.search(header)

        return parse_datetime(match.group("value")) if match else None

    @staticmethod
    def date_from_recorded_field(header: str) -> datetime | None:
        """Date format 5: a `Recorded:` or `Recorded at:` field.

        Args:
            header: Normalized meeting text.

        Returns:
            The recording date and time, or None when there is no readable field.
        """
        date_text = header_field(header, "recorded at|recorded")

        return parse_datetime(date_text) if date_text else None

    @staticmethod
    def date_from_start_field(header: str) -> datetime | None:
        """Date format 6: a `Meeting start:` or `When:` field.

        Args:
            header: Normalized meeting text.

        Returns:
            The meeting date and time, or None when there is no readable field.
        """
        date_text = header_field(header, "meeting start|when")

        return parse_datetime(date_text) if date_text else None

    @staticmethod
    def content_from_python_list(content: str) -> str | None:
        """Content format 1: the meeting is a Python list of strings, one part each.

        Args:
            content: Raw meeting content.

        Returns:
            The parts joined by line breaks, or None when the content is not such a list.
        """
        stripped = content.strip()
        if not stripped.startswith("["):
            return None

        try:
            decoded = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return None

        if not isinstance(decoded, list):
            return None

        return normalize_newlines("\n".join(str(item) for item in decoded))

    @staticmethod
    def content_from_text(content: str) -> str | None:
        """Content format 2: plain text, the usual case.

        Args:
            content: Raw meeting content.

        Returns:
            The text with real line breaks, or None when it is empty.
        """
        return normalize_newlines(content) or None

    @staticmethod
    def attendees_from_labels(header: str) -> list[Person] | None:
        """Attendees format 1: `Attendees:` lists, inline or on the lines below.

        Args:
            header: Normalized meeting text.

        Returns:
            The attendees, or None when there is no labeled list.
        """
        return split_people(attendee_text(header)) or None

    @staticmethod
    def attendees_from_unlabeled_list(header: str) -> list[Person] | None:
        """Attendees format 2: an `Attendees` line with no colon, then one line per group.

        Groups look like `Redwood Inference: Ann (AE), Bob` or `- Ann (Redwood)`.

        Args:
            header: Normalized meeting text.

        Returns:
            The attendees, or None when there is no such list.
        """
        match = UNLABELED_ATTENDEES.search(header)
        if match is None:
            return None

        groups: list[str] = []
        for line in header[match.end() :].lstrip("\n").splitlines():
            if not line.strip() or line.strip().startswith("---"):
                break

            groups.append(GROUP_LABEL.sub("", BULLET.sub("", line)))

        return split_people(", ".join(groups)) or None

    @staticmethod
    def attendees_from_transcript(header: str) -> list[Person] | None:
        """Attendees format 3: the speakers of `[00:12] Name (Role): ...` transcript lines.

        Used when the meeting has no attendee list. Only people who spoke are found;
        placeholders like `Speaker 5`, `Note`, `All` or `Host` and bots are skipped.

        Args:
            header: Normalized meeting text.

        Returns:
            Each speaker once, with the first role given, or None when no line matches.
        """
        roles: dict[str, str | None] = {}
        for line in header.splitlines():
            match = TRANSCRIPT_SPEAKER.match(line)
            if match is None or PLACEHOLDER_SPEAKER.match(match.group("name")):
                continue

            name = match.group("name")
            if roles.get(name) is None:
                roles[name] = match.group("role")

        return [Person(name=name, role=role) for name, role in roles.items()] or None


def meeting_header_line(header: str) -> str | None:
    """Read the `Meeting header:` line, which can start with an unlabeled date.

    Args:
        header: Normalized meeting text.

    Returns:
        The line after the label, or None when there is no such line.
    """
    match = MEETING_HEADER_LINE.search(header)

    return match.group("value").strip() or None if match else None


def header_field(header: str, labels: str) -> str | None:
    """Read a short header field such as `Date: 2027-07-23`.

    Args:
        header: Normalized meeting text.
        labels: Alternative labels, separated by `|`.

    Returns:
        The field value up to the next `;`, `|` or line break, or up to a sentence
        that starts another field (`~46 minutes. Attendees: ...`); None when absent.
    """
    pattern = re.compile(rf"(?im){FIELD_PREFIX}(?:{labels})[ \t]*:[ \t]*(?P<value>[^;|\n]*)")
    match = pattern.search(header)
    if match is None:
        return None

    return NEXT_FIELD.sub("", match.group("value")).strip() or None


def attendee_text(header: str) -> str:
    """Read every attendee list, e.g. `Attendees (Redwood): ...` and `Attendees (Acme): ...`.

    A list is inline after the label, or on the lines below it (bulleted or not).

    Args:
        header: Normalized meeting text.

    Returns:
        All attendee lists joined by commas, or an empty string when absent.
    """
    pattern = re.compile(
        rf"(?im){FIELD_PREFIX}(?:attendees|participants)(?:[ \t]*\([^)\n]*\))?"
        r"[ \t]*:[ \t]*(?P<value>.*)$"
    )
    lists: list[str] = []
    for match in pattern.finditer(header):
        if match.group("value").strip():
            lists.append(match.group("value"))
            continue

        for line in header[match.end() :].lstrip("\n").splitlines():
            if not line.strip() or LABEL_LINE.match(line) and not BULLET.match(line):
                break

            lists.append(BULLET.sub("", line))

    return ", ".join(lists)


def labeled_block(header: str, label: re.Pattern[str]) -> str | None:
    """Read a multi-line block that starts at a label and ends at a blank line or label.

    A transcript line (`[00:00] Name: ...`) also ends the block, for meetings without a
    `transcript:` key; a summary written as an agenda (`0:05 - Discussed ...`) does not.

    Args:
        header: Normalized meeting text.
        label: Pattern of the label line, with an optional `inline` group for
            same-line text.

    Returns:
        The block text, or None when the label is absent or empty.
    """
    match = label.search(header)
    if match is None:
        return None

    inline = (match.group("inline") or "").strip()
    lines = [inline] if inline else []
    for line in header[match.end() :].removeprefix("\n").splitlines():
        stripped = line.strip()
        if not stripped and lines or is_transcript_line(line):
            break

        ends_block = LABEL_LINE.match(line) and not BULLET.match(line)
        if lines and ends_block or stripped.startswith("---"):
            break

        if stripped:
            lines.append(stripped)

    return "\n".join(lines) or None



def is_transcript_line(line: str) -> bool:
    """Tell whether a line is a transcript line: `[00:00] ...` or `00:00 Name: ...`.

    Args:
        line: One line of the meeting.

    Returns:
        True when the line starts with a bracketed timestamp or a timestamped speaker.
    """
    return bool(BRACKETED_TIMESTAMP.match(line) or TRANSCRIPT_SPEAKER.match(line))


def topics(header: str) -> list[str]:
    """Read the topics, separated inline by `;` or `,`, or listed one per line.

    Args:
        header: Normalized meeting text.

    Returns:
        The topics in order.
    """
    block = labeled_block(header, TOPICS_LABEL)
    if block is None:
        return []

    if "\n" in block:
        items = block.splitlines()
    else:
        items = block.split(";") if ";" in block else block.split(",")

    return [BULLET.sub("", item).strip() for item in items if BULLET.sub("", item).strip()]
