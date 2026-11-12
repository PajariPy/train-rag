"""Tests for cleaning a generated question."""

from train_rag.training.questions import clean_question


def test_clean_question() -> None:
    """One clean line comes out; empty or overlong answers are rejected."""
    assert clean_question('"Query: What is X?"\n') == "What is X?"
    assert clean_question("\n**What changed in the rollout?**\nextra line") == (
        "What changed in the rollout?"
    )
    assert clean_question("") is None
    assert clean_question("a" * 500) is None
