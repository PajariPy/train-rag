"""Tests for picking hard negatives below the reranker's biggest score drop."""

from train_rag.training.drop_rule import (
    biggest_drop,
    candidates_with_positive,
    negatives_excluded,
    pick_negatives,
)


def test_biggest_drop_counts_documents_above_it() -> None:
    """The count is the number of scores before the largest step."""
    assert biggest_drop([0.95, 0.80, 0.79, 0.78]) == 1
    assert biggest_drop([0.90, 0.89, 0.70, 0.69]) == 2


def test_pick_negatives_takes_just_below_the_drop() -> None:
    """Negatives are the best-scoring documents right under the drop."""
    scores = {"pos": 0.95, "a": 0.80, "b": 0.79, "c": 0.78, "d": 0.77}

    assert pick_negatives(scores, "pos", set(), count=2) == ["a", "b"]


def test_positive_below_the_drop_skips_the_question() -> None:
    """A source document under the drop means the question is skipped."""
    scores = {"x": 0.95, "pos": 0.80, "a": 0.79}

    assert pick_negatives(scores, "pos", set()) is None


def test_excluded_and_above_positive_never_negatives() -> None:
    """Evaluation documents are never negatives."""
    scores = {"pos": 0.95, "evaldoc": 0.80, "a": 0.79}

    assert pick_negatives(scores, "pos", {"evaldoc"}) == ["a"]


def test_no_negatives_means_skip() -> None:
    """A question with nothing below the drop is skipped."""
    assert pick_negatives({"pos": 0.9}, "pos", set()) is None


def test_positive_is_scored() -> None:
    """The source document joins the candidate list once, even when search missed it."""
    assert candidates_with_positive(["a", "b"], "pos") == ["a", "b", "pos"]
    assert candidates_with_positive(["a", "pos"], "pos") == ["a", "pos"]


def test_other_splits_answers_are_never_negatives() -> None:
    """A dev answer can't be a train negative, nor the other way round."""
    positives = {"train": {"t1", "t2"}, "dev": {"d1"}}

    assert negatives_excluded("train", {"e1"}, positives) == {"e1", "d1"}
    assert negatives_excluded("dev", {"e1"}, positives) == {"e1", "t1", "t2"}
