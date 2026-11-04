"""Tests for removing NUL characters before storing parsed documents."""

from train_rag.repository.parsed_document import JsonValue, without_nul


def test_without_nul_cleans_nested_strings() -> None:
    """NUL is removed from strings at any depth; other values stay."""
    data: JsonValue = {"messages": [{"body": "Thanks Evelyn \x00!"}], "count": 2, "flag": None}

    assert without_nul(data) == {
        "messages": [{"body": "Thanks Evelyn !"}],
        "count": 2,
        "flag": None,
    }
