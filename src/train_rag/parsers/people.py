"""Parse comment threads and lists of people, shared by several parsers."""

import re

from train_rag.models.parsed import Comment, Person
from train_rag.parsers.text import parse_datetime, unique

NAME = r"[A-Z][\w'’.\-]*(?: [A-Z][\w'’.\-]*){0,3}"
PERSON_NAME = re.compile(r"^[A-Z][a-z'’\-]+(?: [A-Z][a-z'’\-]+)+$")
DATE = r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?"
LEADING_DATE = rf"^\s*(?:[-*•]\s*)?(?:\[?(?P<date>{DATE})\]?\s*(?:[-–—|]\s*)?)?"
COMMENT_START = re.compile(
    rf"{LEADING_DATE}"
    rf"(?P<author>Author reply|{NAME})"
    rf"(?:\s*\((?P<role>[^)]{{1,60}})\))?"
    rf"(?:\s*[-–—]?\s*\(?(?P<late_date>{DATE})\)?)?"
    rf":\s*(?P<text>.*)$"
)
PAREN_AUTHOR = re.compile(
    rf"{LEADING_DATE}\((?P<author>{NAME})(?:,\s*(?P<role>[^)]{{1,60}}))?\):\s*(?P<text>.*)$"
)
PERSON_ITEM = re.compile(r"^\s*(?P<name>[^()]+?)\s*(?:\((?P<role>.*)\))?\s*\.?\s*$")
GROUP_SEPARATOR = re.compile(r"\s[-–—|]\s")
INLINE_SPEAKER = re.compile(rf"(?<=[.!?”\"])\s+(?={NAME}(?:\s*\([^)]{{1,60}}\))?:\s)")
ROLE_WORDS = {
    "AE",
    "CSM",
    "Customer",
    "Data",
    "Design",
    "Eng",
    "Engineer",
    "Engineering",
    "Finance",
    "Infra",
    "Lead",
    "Legal",
    "ML",
    "Manager",
    "Ops",
    "Oncall",
    "PM",
    "Platform",
    "Privacy",
    "Product",
    "QA",
    "Review",
    "SE",
    "SRE",
    "Sales",
    "Security",
    "Support",
}
NOT_PEOPLE = {
    "Action",
    "Action items",
    "Answer",
    "Blocker",
    "Blocking",
    "Context",
    "Decision",
    "Environment",
    "Error",
    "Example",
    "Fix",
    "Follow",
    "Goal",
    "Impact",
    "Next",
    "Next steps",
    "Note",
    "Notes",
    "Outcome",
    "Plan",
    "Proposal",
    "Question",
    "Reason",
    "Resolution",
    "Result",
    "Risk",
    "Root cause",
    "Status",
    "Summary",
    "Update",
    "Workaround",
}


def parse_comments(text: str, inline_speakers: bool = False) -> list[Comment]:
    """Parse a comment section into comments.

    Handles the formats found in the data: an optional leading `YYYY-MM-DD[ HH:MM]`
    and dash, then `Author (Role):`, `Role (Author):` or `(Author):`, optionally with a
    trailing date. Lines that do not start a comment continue the previous one.

    Args:
        text: Body of a comments or review section.
        inline_speakers: Also split lines where several speakers follow each other
            (`... done. Maya: thanks`), as in GitHub review conversations.

    Returns:
        The comments in order.
    """
    comments: list[Comment] = []
    for line in text.splitlines():
        segments = INLINE_SPEAKER.split(line) if inline_speakers else [line]
        for segment in segments:
            comment = parse_comment(segment)
            if comment is not None:
                comments.append(comment)
            elif comments and segment.strip():
                comments[-1].text = f"{comments[-1].text}\n{segment.strip()}"

    return comments


def parse_comment(segment: str) -> Comment | None:
    """Parse one comment start, or tell that the segment continues a comment.

    Args:
        segment: One line, or one speaker's part of a line.

    Returns:
        The comment, or None when the segment does not start a comment.
    """
    match = COMMENT_START.match(segment) or PAREN_AUTHOR.match(segment)
    if match is None or match.group("author") in NOT_PEOPLE:
        return None

    author, role = match.group("author"), match.group("role")
    if role and PERSON_NAME.match(role) and author.split()[-1] in ROLE_WORDS:
        author, role = role, author

    groups = match.groupdict()
    stamp = groups.get("date") or groups.get("late_date")

    return Comment(
        author=author,
        role=role,
        posted_at=parse_datetime(stamp) if stamp else None,
        text=match.group("text").strip(),
    )


def comment_participants(comments: list[Comment]) -> list[str]:
    """Collect the distinct comment authors.

    Args:
        comments: Parsed comments.

    Returns:
        Author names in order of first appearance, without the generic PR author reply.
    """
    return unique([comment.author for comment in comments if comment.author != "Author reply"])


def split_people(text: str) -> list[Person]:
    """Split a list like `Ann Lee (CTO, Acme), Bob; Cy (SRE) - Dan` into people.

    Commas and semicolons inside parentheses belong to the role, not the list; a
    dash between groups (`Ann (Redwood) - Bob (Acme)`) also separates people.

    Args:
        text: People separated by commas or semicolons.

    Returns:
        The people with their roles, if given.
    """
    items: list[str] = []
    depth = 0
    current = ""
    for char in text:
        depth += {"(": 1, ")": -1}.get(char, 0)
        if char in ",;" and depth == 0:
            items.append(current)
            current = ""
            continue

        current += char
    items.append(current)

    people: list[Person] = []
    for item in items:
        person = PERSON_ITEM.match(item)
        if person is not None:
            people.append(Person(name=person.group("name"), role=person.group("role")))
            continue

        for part in GROUP_SEPARATOR.split(item):
            person = PERSON_ITEM.match(part)
            if person is not None:
                people.append(Person(name=person.group("name"), role=person.group("role")))

    return people
