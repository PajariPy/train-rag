"""Score every ordering of the dev reranking set: strict and answered recall, and agreement.

- strict@k: the source document (the one the question was written from) is in the top k.
- answered@k: the top k holds the source or a document Gemini rated as answering the question
  fully (`data/labels.json`, label.py); answered+partial@k also counts "partial".
- unlabelled in top 5: candidates in an ordering's top 5 without a label; rate them with
  label.py before trusting its answered@k.
- top-5 agreement: share of the top 5 shared with today's cross-encoder (1,024 tokens).
- coverage: share of questions whose source document is among the candidates (first search).
- reranker strict@k: strict@k counted only on covered questions, i.e. the reranker alone.

Orderings come from orderings.py (cross-encoder, ColBERT, ColBERT top N + cross-encoder).

Run from the main checkout:
    uv run python <workspace>/exploration/reranker_dev/score.py
"""

import json
from collections.abc import Sequence

from orderings import (
    DATA,
    FoundCandidate,
    Ordering,
    Question,
    all_orderings,
    by_stored_score,
    load_labels,
    load_questions,
)

SPLITS = ("all", "dev", "dev_slack")


def hit(top: Sequence[FoundCandidate], rated: dict[str, str], accepted: tuple[str, ...]) -> bool:
    """Check whether the top holds the source or a document rated with an accepted label.

    Args:
        top: The first k candidates of an ordering.
        rated: The question's labels by document id.
        accepted: Labels that count as answering; empty for strict recall.

    Returns:
        True when one of the candidates counts as right.
    """
    return any(
        candidate["is_expected"] or rated.get(candidate["doc_id"]) in accepted for candidate in top
    )


def metrics(
    questions: list[Question],
    labels: dict[str, dict[str, str]],
    ordering: Ordering,
    reference: Ordering,
) -> dict[str, float]:
    """Compute recall at 1 and 5 end to end and given coverage, unlabelled docs, agreement.

    Args:
        questions: The questions to score.
        labels: Gemini labels per question id and document id.
        ordering: The ordering to score.
        reference: Today's cross-encoder ordering, for the agreement.

    Returns:
        Each metric by name; recall in percent.
    """
    counts = {
        "strict@1": 0,
        "answered@1": 0,
        "strict@5": 0,
        "answered@5": 0,
        "answered+partial@5": 0,
    }
    unlabelled = 0
    shared = 0
    for question in questions:
        ranked = ordering(question)
        rated = labels.get(question["qid"], {})
        for depth in (1, 5):
            counts[f"strict@{depth}"] += hit(ranked[:depth], rated, ())
            counts[f"answered@{depth}"] += hit(ranked[:depth], rated, ("full",))
        counts["answered+partial@5"] += hit(ranked[:5], rated, ("full", "partial"))
        unlabelled += sum(
            1
            for candidate in ranked[:5]
            if not candidate["is_expected"] and candidate["doc_id"] not in rated
        )
        reference_top = {candidate["doc_id"] for candidate in reference(question)[:5]}
        shared += len(reference_top & {candidate["doc_id"] for candidate in ranked[:5]})

    total = len(questions)
    result: dict[str, float] = {
        name: round(100 * value / total, 1) for name, value in counts.items()
    }
    result["unlabelled in top 5"] = unlabelled
    result["top-5 agreement"] = round(100 * shared / (5 * total), 1)

    covered = [question for question in questions if question["expected_in_candidates"]]
    result["coverage"] = round(100 * len(covered) / total, 1)
    for depth in (1, 5):
        found = sum(hit(ordering(question)[:depth], {}, ()) for question in covered)
        result[f"reranker strict@{depth}"] = round(100 * found / len(covered), 1)

    return result


def main() -> None:
    """Print a table per split and save it to data/answered_summary.json."""
    questions = load_questions()
    labels = load_labels()
    orderings = all_orderings()
    reference = by_stored_score("rerank_1024")

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for split in SPLITS:
        part = [question for question in questions if split == "all" or question["split"] == split]
        summary[split] = {
            name: metrics(part, labels, ordering, reference) for name, ordering in orderings.items()
        }
        print(f"\n== {split} ({len(part)} questions)")
        for name, values in summary[split].items():
            print(f"  {name:42s} {values}")

    (DATA / "answered_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
