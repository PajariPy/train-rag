"""Search results as the agent reads them: one card per document, with its short id."""

from train_rag.evaluation.field_agent.tools import SNIPPET_CHARS
from train_rag.mined_fields.index import FieldSearch
from train_rag.mined_fields.profiles import field_card, plain_card, source_line

NO_RESULTS = "No documents found: try other words or filters."


def snippet(text: str) -> str:
    """Cut a text to the start a result card shows, whitespace collapsed.

    Args:
        text: The text.

    Returns:
        Its first `SNIPPET_CHARS` characters.
    """
    return " ".join(text.split())[:SNIPPET_CHARS]


def result_card(index: FieldSearch, label: str, doc_id: str, title: str, text: str) -> str:
    """Render one result as the agent sees it, with its short id.

    A mined document shows its kind, channel and mined fields; any other its title.

    Args:
        index: The mined fields.
        label: The document's short id, e.g. "d3".
        doc_id: The document.
        title: Its title, for a document without mined fields.
        text: Its text (or best chunk).

    Returns:
        The card.
    """
    found = index.card(doc_id)
    if found is None:
        return f"[{label}] " + plain_card(title or text.split("\n", 1)[0], snippet(text))

    document, mined = found
    body = field_card(mined, snippet(text))
    where = source_line(document.source_type, document.channel)

    return f"[{label}] ({where}) {body}"


def listing(cards: list[str]) -> str:
    """Join result cards for the agent.

    Args:
        cards: The cards, best first.

    Returns:
        The tool output.
    """
    return "\n".join(cards) if cards else NO_RESULTS
