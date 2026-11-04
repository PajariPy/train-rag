"""Tests for text normalization."""

from train_rag.processing.normalize import decode_string_list, normalize_content, normalize_text


def test_escaped_line_breaks_become_real_ones() -> None:
    """Literal backslash-n sequences, as stored in many documents, become line breaks."""
    assert normalize_text("one\\ntwo\\r\\nthree") == "one\ntwo\nthree"


def test_escaped_unicode_is_decoded_and_broken_halves_dropped() -> None:
    """Escapes become characters; an emoji pair is decoded, then removed like any emoji."""
    assert normalize_text("It\\u2019s live \\ud83d\\ude80 now \\ud83d") == "It's live now"


def test_html_entities_are_unescaped() -> None:
    """HTML entities become the characters they stand for."""
    assert normalize_text("A&amp;B &lt;tag&gt;") == "A&B <tag>"


def test_invisible_and_control_characters_are_removed() -> None:
    """Zero-width, soft hyphen, direction marks, control and broken characters go."""
    text = "zero​width﻿ soft­hyphen ‮rtl‬ bell\x07 bad� pua"

    assert normalize_text(text) == "zerowidth softhyphen rtl bell bad pua"


def test_unicode_spaces_and_quotes_become_plain() -> None:
    """Non-breaking spaces and typographic quotes become their plain forms."""
    assert normalize_text("“quoted” and ‘single’") == "\"quoted\" and 'single'"


def test_emoji_are_removed() -> None:
    """Emoji and pictographic symbols are dropped, with the spaces around them."""
    assert normalize_text("Shipped ✅ today \U0001f680\U0001f389") == "Shipped today"


def test_spacing_is_tidied_but_indentation_kept() -> None:
    """Runs of spaces shrink, trailing spaces and extra blank lines go, indents stay."""
    text = "def f():\n    return  1   \n\n\n\nnext\r\nline"

    assert normalize_text(text) == "def f():\n    return 1\n\nnext\nline"


def test_list_of_strings_becomes_messages_separated_by_blank_lines() -> None:
    """A thread stored as a list literal, in either quote style, becomes plain text."""
    stored = """['From: Ana\\nHi, it\\'s me', "From: Bo\\nIt's fine"]"""

    assert normalize_content(stored) == "From: Ana\nHi, it's me\n\nFrom: Bo\nIt's fine"


def test_text_that_only_looks_like_a_list_is_kept() -> None:
    """Bracketed text that isn't a list of strings stays as it was."""
    for text in ["[draft] notes", "[1, 2]", "['open", "[see below] and [above]"]:
        assert decode_string_list(text) == text
