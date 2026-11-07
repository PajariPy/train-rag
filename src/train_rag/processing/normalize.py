r"""Text normalization: clean raw document text before it is chunked and embedded.

Every rule here fixes something found in the corpus (counts are documents out of
511,962): a Python list of strings instead of plain text (115,000 Gmail threads and a few
hundred other documents, quoted with `'` or `"`), literal `\n` instead of line breaks
(146,870), typographic quotes (70,849), emoji (10,436), literal `\u2019`-style escapes
(2,772), HTML entities (331), carriage returns (260), zero-width characters (205),
control characters (77), Unicode spaces (78) and replacement or private-use characters
(96). Tabs are kept: they indent code.
"""

import ast
import html
import re
import unicodedata
from collections.abc import Sequence

from train_rag.models.chunk import CleanDocument
from train_rag.tables import DocumentRecord

ESCAPED_UNICODE = re.compile(r"\\u([0-9a-fA-F]{4})")
ESCAPED_BREAK = re.compile(r"\\r\\n|\\n|\\r")
INVISIBLE = re.compile(
    "["
    "­"  # soft hyphen
    "​-‍⁠﻿"  # zero-width characters and byte-order mark
    "‎‏‪-‮⁦-⁩"  # text-direction marks
    "︀-️"  # variation selectors (emoji presentation)
    "]"
)
CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
BROKEN = re.compile("[�-]")  # replacement and private-use characters
EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, transport, symbols
    "☀-➿"  # miscellaneous symbols and dingbats
    "⬀-⯿"  # stars, arrows-as-emoji
    "‍⃣"  # emoji joiners and keycaps
    "]"
)
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
INNER_SPACES = re.compile(r"(?<=\S)[ ]{2,}")
TRAILING_SPACES = re.compile(r"[ \t]+\n")
BLANK_LINES = re.compile(r"\n{3,}")


def decode_string_list(text: str) -> str:
    """Turn a document stored as a Python list of strings into plain text.

    Many Gmail threads are stored as the text of a list literal, one string per email,
    written with `'` or `"` quotes as Python chose. The strings are decoded, which also
    turns their escaped line breaks and quotes into real characters, and joined by blank
    lines. Anything else, including text that only looks like a list, is returned as is.

    Args:
        text: Raw document text.

    Returns:
        The joined strings, or the text unchanged.
    """
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return text
    try:
        decoded = ast.literal_eval(stripped)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return text
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        return text

    return "\n\n".join(decoded)


def normalize_content(text: str) -> str:
    """Clean a document's text: decode a stored list of strings, then normalize.

    Args:
        text: Raw document text.

    Returns:
        The cleaned text.
    """
    return normalize_text(decode_string_list(text))


def normalize_text(text: str) -> str:
    """Clean text for embedding without changing what it says.

    Escaped line breaks and Unicode escapes are decoded, HTML entities are unescaped,
    and Unicode is NFKC-normalized (non-breaking and other Unicode spaces become plain
    spaces, full-width letters become normal ones). Invisible characters, control
    characters, broken characters and emoji are removed, and typographic quotes become
    plain ones. Runs of spaces inside a line shrink to one; indentation is kept.

    Args:
        text: Raw text.

    Returns:
        The cleaned text, stripped.
    """
    text = ESCAPED_UNICODE.sub(lambda match: chr(int(match.group(1), 16)), text)
    # Escaped emoji come as surrogate pairs (`🚀`): join them, and turn a lone
    # half into a replacement character, removed below.
    text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    text = ESCAPED_BREAK.sub("\n", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLE.sub("", text)
    text = CONTROL.sub("", text)
    text = BROKEN.sub("", text)
    text = EMOJI.sub("", text)
    text = text.translate(QUOTES)
    text = INNER_SPACES.sub(" ", text)
    text = TRAILING_SPACES.sub("\n", text)
    text = BLANK_LINES.sub("\n\n", text)

    return text.strip()


def normalize_documents(documents: Sequence[DocumentRecord]) -> list[CleanDocument]:
    """Normalize the title and text of raw documents.

    Args:
        documents: Raw documents.

    Returns:
        One cleaned document per stored raw document, in order.
    """
    return [
        CleanDocument(
            document_id=document.id,
            doc_id=document.doc_id,
            title=normalize_text(document.title),
            text=normalize_content(document.content),
        )
        for document in documents
        if document.id is not None
    ]
