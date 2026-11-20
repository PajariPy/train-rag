"""The texts field search compares a query with, and the card the agent sees per result."""

from train_rag.mined_fields.schema import NAME_FIELDS, SOURCE_KINDS
from train_rag.models.mined_fields import FieldProfiles, MinedFields

PROFILES = ("subject", "problem", "names")
# Facts shown on a result card, two values each at most.
CARD_FACTS = ("bug", "fix", "client", "component", "person")
CARD_VALUES = 2


def profiles(mined: MinedFields) -> FieldProfiles:
    """Build the three profile texts of a document.

    Args:
        mined: Its mined fields.

    Returns:
        subject = topic + concepts; problem = bug + cause + fix + question; names = every
        name field. Values are joined with "; ".
    """
    fields = mined.fields

    return FieldProfiles(
        subject="; ".join(fields["topic"] + fields["concept"]),
        problem="; ".join(fields["bug"] + fields["cause"] + fields["fix"] + fields["question"]),
        names="; ".join(value for name in NAME_FIELDS for value in fields[name]),
    )


def source_line(source_type: str, channel: str | None) -> str:
    """Describe where a document comes from, as the agent reads it.

    Args:
        source_type: The document's source.
        channel: Its Slack channel, if any.

    Returns:
        E.g. "chat thread #eng-releases" or "email".
    """
    kind = SOURCE_KINDS.get(source_type, source_type)

    return f"{kind} #{channel}" if channel else kind


def field_card(mined: MinedFields, snippet: str) -> str:
    """One mined document as a search result: topic, areas, key facts and a snippet.

    Args:
        mined: Its mined fields.
        snippet: The start of its text.

    Returns:
        The card's lines after the first, indented under it.
    """
    fields = mined.fields
    topic = (fields["topic"] or ["?"])[0]
    facts = [
        f"{name}: {'; '.join(fields[name][:CARD_VALUES])}" for name in CARD_FACTS if fields[name]
    ]
    lines = [topic, f"   areas: {', '.join(mined.areas)}"]
    if facts:
        lines.append("   " + " | ".join(facts))
    lines.append(f"   text: {snippet}…")

    return "\n".join(lines)


def plain_card(title: str, snippet: str) -> str:
    """A document without mined fields as a search result: title and snippet.

    Args:
        title: Its title.
        snippet: The start of its text.

    Returns:
        The card's lines.
    """
    first_line = " ".join(title.split())[:120] or "(no title)"

    return f"{first_line}\n   (no mined fields)\n   text: {snippet}…"
