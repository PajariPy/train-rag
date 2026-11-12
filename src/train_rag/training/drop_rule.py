"""Pick hard negatives below the reranker's biggest score drop (brainstorm 004).

For one question, the reranker scores its source document and the search's candidates.
Sorted from high to low, the biggest step between neighbours in the top `window` splits a
small cluster above it, which may hold other answers (false negatives), from documents
confidently below it. Negatives come from just below the drop. If the source document
itself isn't above the drop, the question is ambiguous and is skipped.
"""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet

WINDOW = 30
NEGATIVES = 5


def biggest_drop(scores: Sequence[float], window: int = WINDOW) -> int:
    """Count the documents above the biggest drop between neighbouring scores.

    Args:
        scores: Scores sorted from high to low.
        window: How many top documents may sit above the drop.

    Returns:
        How many documents sit above the drop.
    """
    top = list(scores[: window + 1])
    if len(top) < 2:
        return len(top)
    steps = [higher - lower for higher, lower in zip(top, top[1:], strict=False)]

    return steps.index(max(steps)) + 1


def pick_negatives(
    scores: Mapping[str, float],
    positive: str,
    excluded: AbstractSet[str],
    count: int = NEGATIVES,
    window: int = WINDOW,
) -> list[str] | None:
    """Choose hard negatives for a question, or skip it.

    Args:
        scores: Reranker score of the source document and every candidate.
        positive: The source document's id.
        excluded: Documents that may never be negatives (evaluation documents).
        count: Most negatives to return.
        window: How many top documents may sit above the drop.

    Returns:
        Up to `count` document ids just below the drop, best first; None when the source
        document isn't above the drop or no negative is left.
    """
    order = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
    above = biggest_drop([scores[doc_id] for doc_id in order], window)
    if order.index(positive) >= above:
        return None
    negatives = [
        doc_id
        for doc_id in order[above:]
        if doc_id not in excluded and scores[doc_id] < scores[positive]
    ][:count]

    return negatives or None


def candidates_with_positive(fused: Sequence[str], positive: str) -> list[str]:
    """Make sure the source document is scored with the search's candidates.

    Args:
        fused: The search's candidate doc ids, best first.
        positive: The source document's id.

    Returns:
        The candidates, with the source document appended if search missed it.
    """
    return list(fused) if positive in fused else [*fused, positive]


def negatives_excluded(
    split: str, evaluation: AbstractSet[str], positives: Mapping[str, AbstractSet[str]]
) -> set[str]:
    """Documents that may not be negatives for one split's questions.

    Evaluation documents never are, and neither are the other splits' answers: pushing a dev
    answer away in training would bias the dev score.

    Args:
        split: The split the questions belong to.
        evaluation: The benchmark's right documents.
        positives: split -> the source documents of its questions.

    Returns:
        The excluded doc ids.
    """
    others = (doc_ids for name, doc_ids in positives.items() if name != split)

    return set(evaluation).union(*others)
