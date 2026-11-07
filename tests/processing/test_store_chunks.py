"""Tests for linking stored chunks to their neighbours."""

from train_rag.repository.embedding import neighbour_ids


def test_neighbours_stay_within_one_document() -> None:
    """Chunks link to the chunks around them, never across a document boundary."""
    owners = [7, 7, 7, 9, 12, 12]
    ids = [101, 102, 103, 104, 105, 106]

    assert neighbour_ids(owners, ids) == [
        (None, 102),
        (101, 103),
        (102, None),
        (None, None),
        (None, 106),
        (105, None),
    ]


def test_no_chunks_give_no_neighbours() -> None:
    """An empty batch links nothing."""
    assert neighbour_ids([], []) == []
