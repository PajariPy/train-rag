"""Documents that follow none of their source's formats."""

from train_rag.models.parsed.base import ParsedDocument


class OpenText(ParsedDocument):
    """A document in none of its source's formats, kept as one block of open text.

    Every parser falls back to it the same way, so these documents get the same
    preprocessing later whatever their source. It is never flagged: there are no
    fields to miss. Text under `MIN_USEFUL_CHARS` characters is marked not useful.
    """

    text: str
