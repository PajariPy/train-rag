"""Recursive chunking: cut a document at its natural breaks into chunks of a size limit."""

from collections.abc import Sequence

from train_rag.models.chunk import Chunk, CleanDocument

SEPARATORS = ("\n\n", "\n", ". ", " ")


class RecursiveChunker:
    """Splits text at the coarsest break that gives small enough pieces.

    Text is cut at paragraph breaks first; a paragraph that is still too long is cut at
    line breaks, then at sentence ends, then between words, and only as a last resort in
    the middle of a word. Neighbouring pieces are joined back up to `size` characters.

    Each chunk is then given a little context: the end of the chunk before it on top and
    the start of the chunk after it at the bottom, so a chunk that starts mid-topic still
    says what the topic is.
    """

    def __init__(self, size: int, context: int, separators: Sequence[str] = SEPARATORS) -> None:
        """Set the limits.

        Args:
            size: Maximum characters of a chunk, without its context.
            context: Characters borrowed from each neighbouring chunk.
            separators: Breaks to cut at, coarsest first.
        """
        self.size = size
        self.context = context
        self.separators = tuple(separators)

    def split(self, text: str) -> list[str]:
        """Cut text into chunks of at most `size` characters.

        Args:
            text: Text to cut.

        Returns:
            The chunks in order, stripped, without empty ones.
        """
        return [chunk for piece in self._split(text, 0) if (chunk := piece.strip())]

    def chunks(self, title: str, text: str) -> list[str]:
        """Cut a document into chunks with the title and neighbouring context added.

        Args:
            title: Document title, put on top of every chunk.
            text: Document text.

        Returns:
            The texts to embed, one per chunk, in order.
        """
        cores = self.split(text) or [""]
        texts = []
        for index, core in enumerate(cores):
            before = tail(cores[index - 1], self.context) if index > 0 else ""
            after = head(cores[index + 1], self.context) if index + 1 < len(cores) else ""
            texts.append("\n".join(part for part in (title, before, core, after) if part))

        return texts

    def _split(self, text: str, level: int) -> list[str]:
        """Cut text at the separator of one level, recursing into pieces that are too long.

        Args:
            text: Text to cut.
            level: Index of the separator to try first.

        Returns:
            Pieces of at most `size` characters.
        """
        if len(text) <= self.size:
            return [text]
        if level == len(self.separators):
            return [text[start : start + self.size] for start in range(0, len(text), self.size)]

        separator = self.separators[level]
        pieces = []
        for part in text.split(separator):
            if len(part) > self.size:
                pieces.extend(self._split(part, level + 1))
            else:
                pieces.append(part)

        return merge(pieces, separator, self.size)


def chunk_documents(chunker: RecursiveChunker, documents: Sequence[CleanDocument]) -> list[Chunk]:
    """Cut cleaned documents into chunks, keeping each chunk's document and position.

    Args:
        chunker: Chunker sized for the embedding model.
        documents: Cleaned documents.

    Returns:
        The chunks, document by document and in order within each.
    """
    return [
        Chunk(
            document_id=document.document_id, doc_id=document.doc_id, position=position, text=text
        )
        for document in documents
        for position, text in enumerate(chunker.chunks(document.title, document.text))
    ]


def merge(pieces: Sequence[str], separator: str, size: int) -> list[str]:
    """Join neighbouring pieces while they fit in `size` characters.

    Args:
        pieces: Pieces in order, each at most `size` characters.
        separator: Text put back between joined pieces.
        size: Maximum characters of a joined chunk.

    Returns:
        The joined chunks.
    """
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        joined = f"{current}{separator}{piece}" if current else piece
        if len(joined) <= size:
            current = joined
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)

    return chunks


def head(text: str, limit: int) -> str:
    """Take the start of a text, cut back to the last whole word.

    Args:
        text: Text to take from.
        limit: Maximum characters.

    Returns:
        At most `limit` characters from the start.
    """
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")

    return cut[:space] if space > 0 else cut


def tail(text: str, limit: int) -> str:
    """Take the end of a text, starting at a whole word.

    Args:
        text: Text to take from.
        limit: Maximum characters.

    Returns:
        At most `limit` characters from the end.
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    space = cut.find(" ")

    return cut[space + 1 :] if space >= 0 else cut
