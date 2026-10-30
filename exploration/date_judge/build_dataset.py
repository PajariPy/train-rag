"""Build a labelled set of Fireflies meetings for evaluating the date judge.

Meetings come from Postgres and are ones whose date the parser found, so the true date
is known. That date is removed: header lines with a date, and every line that repeats
it. Two groups are built:

- 50 synthetic cases from meetings with no other date left in the text. 25 get a clue
  that pins the meeting date hidden in one of five ways. The other 25 do not reveal
  it: 15 have no date at all, and 10 mention a date that says nothing about when the
  meeting was held (an event years ago, an expiry years ahead).
- 20 real cases from meetings whose text still has its own dates (mostly action item
  due dates). These measure how close the judge gets on clues as they occur in the data.

Run: uv run python exploration/date_judge/build_dataset.py
"""

import json
import random
import re
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

from sqlmodel import Session

from train_rag.database import get_engine
from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import FirefliesMeeting
from train_rag.parsers import FirefliesParser
from train_rag.parsers.text import normalize_newlines
from train_rag.repository.document import load_artifacts

DATASET = "enterprise_rag_bench"
DATASET_FILE = Path(__file__).parent / "data" / "dataset.jsonl"
POSITIVES = 25
NEGATIVES = 25
DISTRACTED_NEGATIVES = 10
REAL_CASES = 20
MIN_REAL_DATES = 2
HEAD_LINES = 15
SEED = 7
MIN_CHARS = 3000
MAX_CHARS = 12000
ISO_DATE = re.compile(r"\b(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b")
WRITTEN_DATE = re.compile(
    r"(?i)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?[\s_-]*\d{1,2}(?:st|nd|rd|th)?\b"
)
SLASH_DATE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
COMPACT_DATE = re.compile(r"(?<!\d)20[2-3]\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?!\d)")
YEAR = re.compile(r"\b20[2-3]\d\b")
TRANSCRIPT_LINE = re.compile(r"^\s*[\[(]?\d{1,2}:\d{2}")

Placement = Callable[[str, date, random.Random], str]


def date_like(text: str) -> bool:
    """Tell whether text holds anything that looks like a date or a year.

    Includes compact stamps like `20260610` and `july12` in file names and links.

    Args:
        text: Any text.

    Returns:
        True when a date, a month and day, a slash date or a year appears.
    """
    return any(
        pattern.search(text) for pattern in (ISO_DATE, WRITTEN_DATE, SLASH_DATE, COMPACT_DATE, YEAR)
    )


def date_forms(held_on: date) -> list[str]:
    """Spell a date the ways transcripts write it.

    Args:
        held_on: The meeting date.

    Returns:
        The ISO, written and US slash forms.
    """
    return [
        held_on.isoformat(),
        f"{held_on:%B} {held_on.day}, {held_on.year}",
        f"{held_on:%b} {held_on.day}, {held_on.year}",
        f"{held_on.day} {held_on:%B} {held_on.year}",
        f"{held_on.month:02d}/{held_on.day:02d}/{held_on.year}",
    ]


def strip_meeting_date(content: str, held_on: date) -> str:
    """Remove the meeting date: header lines with a date and lines that repeat it.

    Args:
        content: Meeting text.
        held_on: The meeting date.

    Returns:
        The text without the lines that give the date away.
    """
    forms = date_forms(held_on)
    kept = [
        line
        for index, line in enumerate(normalize_newlines(content).splitlines())
        if not (index < HEAD_LINES and date_like(line)) and not any(form in line for form in forms)
    ]

    return "\n".join(kept)


def transcript_lines(lines: list[str]) -> list[int]:
    """Find the lines that start a transcript turn.

    Args:
        lines: Meeting lines.

    Returns:
        Indexes of lines that start with a timestamp, or of every line when none do.
    """
    starts = [index for index, line in enumerate(lines) if TRANSCRIPT_LINE.match(line)]

    return starts or list(range(len(lines)))


def append_to_turn(content: str, remark: str, position: float) -> str:
    """Append a remark to a transcript turn at a relative position in the meeting.

    Args:
        content: Meeting text.
        remark: Text to append.
        position: 0 for the first turn, 1 for the last.

    Returns:
        The text with the remark added.
    """
    lines = content.splitlines()
    turns = transcript_lines(lines)
    index = turns[min(int(position * len(turns)), len(turns) - 1)]
    lines[index] = lines[index] + remark

    return "\n".join(lines)


def today_statement(content: str, held_on: date, rng: random.Random) -> str:
    """Someone states today's date in the first turn.

    Args:
        content: Meeting text.
        held_on: The meeting date.
        rng: Random source, unused here.

    Returns:
        The text with the clue.
    """
    day = f"{held_on:%A, %B} {held_on.day}, {held_on.year}"
    remark = f" Quick note for the recording: today is {day}."

    return append_to_turn(content, remark, 0.0)


def relative_to_event(content: str, held_on: date, rng: random.Random) -> str:
    """A turn ties a dated event to the meeting day: it happened a few days ago.

    Args:
        content: Meeting text.
        held_on: The meeting date.
        rng: Random source for the gap.

    Returns:
        The text with the clue.
    """
    days = rng.choice([2, 3])
    event = held_on - timedelta(days=days)
    remark = f" (the hotfix went out on {event.isoformat()}, so {days} days ago now)"

    return append_to_turn(content, remark, 0.4)


def deadline_from_today(content: str, held_on: date, rng: random.Random) -> str:
    """The last turn sets a deadline a week from today.

    Args:
        content: Meeting text.
        held_on: The meeting date.
        rng: Random source, unused here.

    Returns:
        The text with the clue.
    """
    deadline = held_on + timedelta(days=7)
    remark = f" I'll have the recap out by {deadline.isoformat()}, exactly one week from today."

    return append_to_turn(content, remark, 1.0)


def week_of_banner(content: str, held_on: date, rng: random.Random) -> str:
    """A banner above the notes names the week of the meeting.

    Args:
        content: Meeting text.
        held_on: The meeting date.
        rng: Random source, unused here.

    Returns:
        The text with the clue.
    """
    monday = held_on - timedelta(days=held_on.weekday())

    return f"Weekly sync notes — week of {monday:%B} {monday.day}, {monday.year}\n\n{content}"


def stamped_file(content: str, held_on: date, rng: random.Random) -> str:
    """A turn shares a file stamped with today's date.

    Args:
        content: Meeting text.
        held_on: The meeting date.
        rng: Random source, unused here.

    Returns:
        The text with the clue.
    """
    stamp = held_on.strftime("%Y%m%d")
    path = f"s3://meetings/recordings/sync-{stamp}.mp4"
    remark = f" I'm saving today's recording as {path} for later."

    return append_to_turn(content, remark, 0.6)


def unrelated_date(content: str, held_on: date, rng: random.Random) -> str:
    """Mention a date that does not place the meeting: years ago or years ahead.

    Args:
        content: Meeting text.
        held_on: The meeting date, used only to keep the mention far from it.
        rng: Random source for the wording and the date.

    Returns:
        The text with a distracting date.
    """
    past = date(held_on.year - rng.choice([3, 4]), rng.randint(1, 12), rng.randint(1, 28))
    future = date(held_on.year + rng.choice([4, 5]), rng.randint(1, 12), rng.randint(1, 28))
    remark = rng.choice(
        [
            f" We hit something similar back on {past.isoformat()}, long before this team existed.",
            f" Their signing certificate is valid until {future.isoformat()}, so no rush there.",
            f" The original master agreement was signed {past.isoformat()} and auto-renews yearly.",
        ]
    )

    return append_to_turn(content, remark, rng.random())


PLACEMENTS: dict[str, Placement] = {
    "today_statement": today_statement,
    "relative_to_event": relative_to_event,
    "deadline_from_today": deadline_from_today,
    "week_of_banner": week_of_banner,
    "stamped_file": stamped_file,
}


def dated_meetings(session: Session) -> list[tuple[Artifact, FirefliesMeeting]]:
    """Load meetings whose date the parser found and whose title has no date.

    Args:
        session: Database session.

    Returns:
        Meetings of a moderate length with their parsed result.
    """
    parser = FirefliesParser()
    meetings = []
    for artifact in load_artifacts(session, DATASET, SourceType.FIREFLIES):
        meeting = parser.parse(artifact)
        if meeting.held_on is None or date_like(artifact.title):
            continue
        if MIN_CHARS <= len(artifact.content) <= MAX_CHARS:
            meetings.append((artifact, meeting))

    return meetings


def main() -> None:
    """Build both groups and write them as JSON lines."""
    rng = random.Random(SEED)
    parser = FirefliesParser()
    with Session(get_engine()) as session:
        meetings = dated_meetings(session)
    rng.shuffle(meetings)

    clean: list[tuple[Artifact, date, str]] = []
    real: list[tuple[Artifact, date, str]] = []
    for artifact, meeting in meetings:
        held_on = meeting.held_on
        if held_on is None:
            continue
        stripped = strip_meeting_date(artifact.content, held_on)
        if parser.parse(artifact.model_copy(update={"content": stripped})).held_on is not None:
            continue
        if not date_like(stripped):
            clean.append((artifact, held_on, stripped))
        elif len(ISO_DATE.findall(stripped)) >= MIN_REAL_DATES:
            real.append((artifact, held_on, stripped))

    rows = []
    placements = list(PLACEMENTS)
    for index, (artifact, held_on, text) in enumerate(clean[: POSITIVES + NEGATIVES]):
        positive = index < POSITIVES
        if positive:
            placement = placements[index % len(placements)]
            content = PLACEMENTS[placement](text, held_on, rng)
        elif index < POSITIVES + DISTRACTED_NEGATIVES:
            placement = "unrelated_date"
            content = unrelated_date(text, held_on, rng)
        else:
            placement = "no_date"
            content = text
        rows.append((artifact, content, positive, held_on, placement, "synthetic"))
    for artifact, held_on, text in real[:REAL_CASES]:
        rows.append((artifact, text, True, held_on, "real_dates_in_text", "real"))

    lines = [
        json.dumps(
            {
                "id": f"case_{index + 1:02d}",
                "group": group,
                "document_id": artifact.document_id,
                "title": artifact.title,
                "content": content,
                "has_date": has_date,
                "held_on": held_on.isoformat() if has_date else None,
                "true_date": held_on.isoformat(),
                "placement": placement,
            },
            ensure_ascii=False,
        )
        + "\n"
        for index, (artifact, content, has_date, held_on, placement, group) in enumerate(rows)
    ]
    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATASET_FILE.write_text("".join(lines))
    print(
        f"Wrote {len(rows)} cases to {DATASET_FILE}: from {len(clean)} meetings with no date "
        f"left and {len(real)} with their own dates left."
    )


if __name__ == "__main__":
    main()
