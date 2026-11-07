"""Tests for the keyword search snippets."""

from train_rag.evaluation.snippets import (
    LONG_LINE,
    MAX_SNIPPETS,
    query_terms,
    snippet_context,
    snippets,
)
from train_rag.models import SourceType
from train_rag.models.retrieval import SearchHit

LINES = [f"filler line {number}" for number in range(20)]


def test_query_terms_are_stemmed_without_stopwords() -> None:
    """The query is reduced to the stems BM25 matches, without stopwords."""
    terms = query_terms("What is the pass rate for machines?")

    assert terms == {"what", "pass", "rate", "machin"}


def test_matching_line_comes_with_two_lines_around_it() -> None:
    """A matching short line is shown with two lines before and after it."""
    lines = [*LINES]
    lines[10] = "the default pass rate is 0.92"

    assert snippets("\n".join(lines), query_terms("pass rate")) == ["\n".join(lines[8:13])]


def test_lines_holding_more_query_words_win() -> None:
    """Lines with more different query words come first; ties go to the earlier line."""
    lines = [*LINES]
    lines[2] = "pass"
    lines[15] = "pass rate threshold"
    lines[18] = "rate"
    result = snippets("\n".join(lines), query_terms("pass rate threshold"))

    assert len(result) == MAX_SNIPPETS
    assert "pass rate threshold" in result[1]
    assert result[0].startswith("filler line 0")


def test_a_line_inside_another_snippet_is_not_shown_twice() -> None:
    """A matching line already shown as context is skipped for the next match."""
    lines = [*LINES]
    lines[10] = "pass rate"
    lines[11] = "pass"
    lines[19] = "rate"
    result = snippets("\n".join(lines), query_terms("pass rate"))

    assert len(result) == 2
    assert "filler line 17" in result[1]


def test_long_line_is_cut_to_a_window_around_the_match() -> None:
    """A line longer than LONG_LINE is shown as a window around its first match."""
    text = "x " * 400 + "kernel_stability_threshold=0.92 " + "y " * 400
    (passage,) = snippets(text, query_terms("kernel_stability_threshold"))

    assert len(text) > LONG_LINE
    assert passage.startswith("…")
    assert passage.endswith("…")
    assert "kernel_stability_threshold" in passage
    assert len(passage) < 520


def test_no_match_falls_back_to_the_start() -> None:
    """Without a matching line, the start of the document is shown."""
    text = "Title only\nnothing here"

    assert snippets(text, query_terms("pass rate")) == [text]


def test_context_names_each_document_by_id_with_its_snippets() -> None:
    """Each result shows its number, source, title and id, then only its snippets."""
    hit = SearchHit(
        doc_id="doc-7",
        document_id=7,
        source_type=SourceType.SLACK,
        field="document",
        text="eng-runtime\n" + "\n".join(LINES) + "\nthe pass rate dropped",
        distance=0.0,
    )
    context = snippet_context([hit], "pass rate")

    assert context.startswith("[1] slack: eng-runtime (id: doc-7, ")
    assert "the pass rate dropped" in context
    assert "filler line 3" not in context
