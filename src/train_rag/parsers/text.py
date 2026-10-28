"""Text helpers shared by the parsers."""

import re
from datetime import datetime

SECTION_KEY = re.compile(r"^([a-z][a-z0-9_]*):[ \t]*(.*)$")
ISO_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")
WRITTEN_DATE = re.compile(r"(?P<month>[A-Z][a-z]{2,8})\.? (?P<day>\d{1,2}),? (?P<year>\d{4})")
LITERAL_NEWLINE = "\\n"


def normalize_newlines(text: str) -> str:
    """Turn literal backslash-n sequences into real line breaks.

    Some documents were exported with escaped line breaks; they are converted only
    when they outnumber the real ones, so code snippets keep their escapes.

    Args:
        text: Raw document content.

    Returns:
        The content with real line breaks.
    """
    if text.count(LITERAL_NEWLINE) <= text.count("\n"):
        return text

    return text.replace(LITERAL_NEWLINE, "\n")


def split_sections(text: str) -> dict[str, str]:
    """Split flattened `key:` sections, the export format of tickets and CRM records.

    A section starts at a line holding a lowercase snake_case key followed by a colon
    and runs until the next such line.

    Args:
        text: Normalized document content.

    Returns:
        Section bodies keyed by section name, in document order.
    """
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        match = SECTION_KEY.match(line)
        if match is None:
            if current is not None:
                lines.append(line)
            continue

        if current is not None:
            sections.setdefault(current, "\n".join(lines).strip())
        current = match.group(1)
        lines = [match.group(2)] if match.group(2) else []

    if current is not None:
        sections.setdefault(current, "\n".join(lines).strip())

    return sections


def section(text: str, key: str) -> str | None:
    """Read one `key:` section.

    Args:
        text: Normalized document content.
        key: Section name, e.g. `description`.

    Returns:
        The section text, or None when the section is absent or empty.
    """
    return split_sections(text).get(key) or None


def leading_text(text: str) -> str | None:
    """Read the prose before the first `key:` section.

    Covers documents written without section keys, often under capitalized labels
    like `Summary:` or `## Context` instead.

    Args:
        text: Normalized document content.

    Returns:
        The text before the first section, or None when the document starts with one.
    """
    lines: list[str] = []
    for line in text.splitlines():
        if SECTION_KEY.match(line):
            break

        lines.append(line)

    return "\n".join(lines).strip() or None


def parse_datetime(text: str) -> datetime | None:
    """Parse the first date in a text, ISO (`2026-03-10 09:15`) or written (`Dec 12, 2025`).

    Args:
        text: Text that may contain a date.

    Returns:
        The parsed value, or None when no valid date is present.
    """
    iso = ISO_DATETIME.search(text)
    if iso is not None:
        try:
            return datetime.fromisoformat(iso.group(0))
        except ValueError:
            return None

    written = WRITTEN_DATE.search(text)
    if written is None:
        return None

    month = written.group("month")[:3]
    try:
        return datetime.strptime(f"{month} {written['day']} {written['year']}", "%b %d %Y")
    except ValueError:
        return None


def unique(values: list[str]) -> list[str]:
    """Drop duplicates while keeping the first-seen order.

    Args:
        values: Values that may repeat.

    Returns:
        The distinct values in order.
    """
    return list(dict.fromkeys(values))


def labeled_value(
    text: str, label: str, stop_labels: list[str], ignore_case: bool = False
) -> str | None:
    """Read the value after an inline `Label:` up to the next known label.

    Used for prose like `Issue summary: ... Impact: ... Environment: ...`, where
    several labeled fields share one paragraph.

    Args:
        text: Text containing the label.
        label: Label whose value to read.
        stop_labels: Labels that end the value when they follow it.
        ignore_case: Also match the labels in another case, e.g. `Issue Summary:`.

    Returns:
        The trimmed value, or None when the label is absent or empty.
    """
    stops = "|".join(re.escape(stop) for stop in sorted(stop_labels, key=len, reverse=True))
    pattern = re.compile(
        rf"(?<![\w-]){re.escape(label)}:\s*(?P<value>.*?)(?=\s+(?:{stops}):|\n\s*\n|\Z)",
        re.DOTALL | (re.IGNORECASE if ignore_case else 0),
    )
    match = pattern.search(text)
    if match is None:
        return None

    return match.group("value").strip() or None
