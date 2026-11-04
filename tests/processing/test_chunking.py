"""Tests for the recursive chunker."""

from train_rag.processing.chunking import RecursiveChunker, head, tail


def test_short_text_stays_one_chunk() -> None:
    """Text within the size limit is not cut."""
    assert RecursiveChunker(size=100, context=10).split("One short paragraph.") == [
        "One short paragraph."
    ]


def test_cuts_at_paragraphs_before_lines() -> None:
    """Paragraphs are kept whole when each fits, and joined while they fit together."""
    text = "aaaa aaaa\nbbbb\n\ncccc cccc\n\ndddd dddd"

    assert RecursiveChunker(size=20, context=0).split(text) == [
        "aaaa aaaa\nbbbb",
        "cccc cccc\n\ndddd dddd",
    ]


def test_long_paragraph_falls_back_to_finer_breaks() -> None:
    """A paragraph over the limit is cut at sentence ends, then between words."""
    text = "First sentence here. Second sentence here. " + "word " * 10
    chunks = RecursiveChunker(size=25, context=0).split(text)

    assert chunks[:2] == ["First sentence here", "Second sentence here"]
    assert all(len(chunk) <= 25 for chunk in chunks)
    assert " ".join(chunks).split() == text.replace(". ", " ").split()


def test_word_longer_than_size_is_cut_hard() -> None:
    """A single word over the limit is cut into pieces of the limit."""
    assert RecursiveChunker(size=4, context=0).split("abcdefghij") == ["abcd", "efgh", "ij"]


def test_chunks_add_title_and_neighbouring_context() -> None:
    """Each chunk gets the title, the end of the chunk before and the start of the next."""
    text = "alpha beta gamma\n\ndelta epsilon zeta\n\neta theta iota"
    chunks = RecursiveChunker(size=20, context=10).chunks("Title", text)

    assert chunks == [
        "Title\nalpha beta gamma\ndelta",
        "Title\ngamma\ndelta epsilon zeta\neta theta",
        "Title\nzeta\neta theta iota",
    ]


def test_empty_document_gives_one_chunk_with_the_title() -> None:
    """A document without text is still searchable by its title."""
    assert RecursiveChunker(size=20, context=5).chunks("Only a title", "") == ["Only a title"]


def test_head_and_tail_keep_whole_words() -> None:
    """Context taken from a neighbour never starts or ends mid-word."""
    assert head("alpha beta gamma", 12) == "alpha beta"
    assert tail("alpha beta gamma", 12) == "beta gamma"


def test_zero_context_takes_nothing_from_neighbours() -> None:
    """With no context, a chunk is only its title and its own text."""
    assert tail("alpha beta", 0) == ""
    assert head("alpha beta", 0) == ""
