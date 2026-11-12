"""Tests for splitting training questions across types and sources."""

from train_rag.training.quotas import TYPE_WEIGHTS, is_usable_source, split_by_weight


def test_split_by_weight_sums_and_rounds() -> None:
    """Counts follow the weights and always sum to the total."""
    assert split_by_weight(2600, TYPE_WEIGHTS) == {
        "basic": 1264,
        "semantic": 903,
        "intra_document_reasoning": 289,
        "miscellaneous": 144,
    }
    assert split_by_weight(10, {"a": 1, "b": 0}) == {"a": 10, "b": 0}


def test_short_documents_are_not_sources() -> None:
    """Too-short documents can't be sources; intra-document needs 3,000 characters."""
    assert not is_usable_source(150, "basic")
    assert is_usable_source(500, "basic")
    assert not is_usable_source(2999, "intra_document_reasoning")
    assert is_usable_source(3000, "intra_document_reasoning")
