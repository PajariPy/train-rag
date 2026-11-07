"""Short previews of keyword search results: the lines that match, with a little context.

The agent's keyword search shows each found document as a few snippets instead of its full
text, so it can scan many results for few tokens and read in full only the ones that look
right. A line matches when it holds one of the query's words as BM25 sees them: lowercased,
without stopwords, stemmed.
"""

import re
from collections.abc import Sequence

import Stemmer

from train_rag.index_bm25 import tokenize
from train_rag.models.retrieval import SearchHit

MAX_SNIPPETS = 2  # snippets per document
CONTEXT_LINES = 2  # lines shown before and after a matching line
LONG_LINE = 500  # a longer line is cut to a window around its first match
WINDOW_CHARS = 250  # characters shown on each side of that match
FALLBACK_CHARS = 300  # shown from the start when no line matches (the title did)
WORD = re.compile(r"(?u)\b\w\w+\b")  # bm25s's own word pattern

STEMMER = Stemmer.Stemmer("english")


def query_terms(query: str) -> set[str]:
    """The query's words as BM25 matches them.

    Args:
        query: The keyword query.

    Returns:
        Its stemmed words, without stopwords.
    """
    return set(tokenize([query], STEMMER)[0])


def matches(line: str, terms: set[str]) -> list[tuple[int, str]]:
    """Where a line holds the query's words.

    Args:
        line: One line of a document.
        terms: The query's stemmed words.

    Returns:
        The start offset and stem of every matching word, in order.
    """
    found = []
    for word in WORD.finditer(line):
        stem = STEMMER.stemWord(word.group().lower())
        if stem in terms:
            found.append((word.start(), stem))

    return found


def snippets(text: str, terms: set[str]) -> list[str]:
    """The best matching passages of a document.

    Lines are ranked by how many different query words they hold, then by position. Each
    chosen line is shown with `CONTEXT_LINES` around it, or, when it is longer than
    `LONG_LINE`, as a window of `WINDOW_CHARS` around its first match. A line already
    shown as another snippet's context is not chosen again.

    Args:
        text: The document's full text.
        terms: The query's stemmed words.

    Returns:
        Up to `MAX_SNIPPETS` passages, in document order.
    """
    lines = text.splitlines()
    scored = []
    for index, line in enumerate(lines):
        found = matches(line, terms)
        if found:
            scored.append((-len({stem for _, stem in found}), index, found))
    scored.sort()

    chosen: list[tuple[int, str]] = []
    covered: set[int] = set()
    for _, index, found in scored:
        if len(chosen) == MAX_SNIPPETS:
            break
        if index in covered:
            continue
        line = lines[index]
        if len(line) > LONG_LINE:
            start = max(0, found[0][0] - WINDOW_CHARS)
            end = min(len(line), found[0][0] + WINDOW_CHARS)
            passage = ("…" if start else "") + line[start:end] + ("…" if end < len(line) else "")
            covered.add(index)
        else:
            first = max(0, index - CONTEXT_LINES)
            last = min(len(lines), index + CONTEXT_LINES + 1)
            passage = "\n".join(lines[first:last]).strip()
            covered.update(range(first, last))
        chosen.append((index, passage))
    if not chosen:
        return [text[:FALLBACK_CHARS].strip() + ("…" if len(text) > FALLBACK_CHARS else "")]

    return [passage for _, passage in sorted(chosen)]


def snippet_context(hits: Sequence[SearchHit], query: str) -> str:
    """Lay out keyword search results as snippets for the agent.

    Args:
        hits: The found documents, best first, each with its full text (title on top).
        query: The keyword query.

    Returns:
        One block per document: its number, source, title, id and length, then its
        snippets.
    """
    terms = query_terms(query)
    blocks = []
    for number, hit in enumerate(hits, 1):
        title = hit.text.splitlines()[0] if hit.text else ""
        passages = "\n  ...\n".join(
            "  " + passage.replace("\n", "\n  ") for passage in snippets(hit.text, terms)
        )
        blocks.append(
            f"[{number}] {hit.source_type}: {title} (id: {hit.doc_id}, "
            f"{len(hit.text):,} characters)\n{passages}"
        )

    return "\n\n".join(blocks)
