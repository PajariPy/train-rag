"""Score LateOn on deeper candidates: does handing it gte + BM25 top 200 raise accuracy?

Everything is end to end over all questions (a question whose right document was never found
counts as a miss), so the depths compare fairly with today:

- coverage: the source document is among the candidates at that depth.
- total@k: the source document is in the top k.
- rerank@5: total@5 counted only on covered questions, i.e. the reranker alone.
- ans@k: the top k holds the source or a document Gemini rated "full" (`data/labels.json`);
  unlabelled in top 5 says how many top-5 pairs still need a rating (label.py).

Rows: today's cross-encoder and LateOn on ~97 candidates (the original set), then each deep run
(`data/deep_scores_<run>.json`, colbert.py --deep) at 50 / 100 / 150 / 200 per search.

Run from the main checkout:
    uv run python <workspace>/exploration/reranker_dev/deep_score.py
"""

import json

from orderings import (
    DATA,
    FoundCandidate,
    Question,
    Ranker,
    all_orderings,
    deep_orderings,
    load_deep_found,
    load_labels,
    load_questions,
    within,
)
from score import hit

SPLITS = ("all", "dev", "dev_slack")
BASELINES = ("cross-encoder 1,024", "colbert lateon")


def covered(question: Question, name: str, found: dict[str, list[FoundCandidate]]) -> bool:
    """Check whether the source document is among a row's candidates.

    Args:
        question: The question.
        name: The row's ordering name ("deep <run> @<depth>" for deep rows).
        found: Deep candidates per question id.

    Returns:
        True when the right document can be ranked at all.
    """
    if not name.startswith("deep "):
        return question["expected_in_candidates"]

    depth = int(name.rsplit("@", 1)[1])

    return any(cand["is_expected"] and within(cand, depth) for cand in found[question["qid"]])


def metrics(
    questions: list[Question],
    labels: dict[str, dict[str, str]],
    name: str,
    ordering: Ranker,
    found: dict[str, list[FoundCandidate]],
) -> dict[str, float]:
    """Compute coverage, total@1/5, rerank@5, ans@1/5 and unlabelled top-5 pairs for a row.

    Args:
        questions: The questions to score.
        labels: Gemini labels per question id and document id.
        name: The row's ordering name.
        ordering: The row's ordering.
        found: Deep candidates per question id.

    Returns:
        Each metric by name; shares in percent.
    """
    counts = {"total@1": 0, "total@5": 0, "ans@1": 0, "ans@5": 0}
    covered_count = 0
    reranked = 0
    unlabelled = 0
    for question in questions:
        ranked = ordering(question)
        rated = labels.get(question["qid"], {})
        for depth in (1, 5):
            counts[f"total@{depth}"] += hit(ranked[:depth], {}, ())
            counts[f"ans@{depth}"] += hit(ranked[:depth], rated, ("full",))

        unlabelled += sum(
            1 for cand in ranked[:5] if not cand["is_expected"] and cand["doc_id"] not in rated
        )
        if covered(question, name, found):
            covered_count += 1
            reranked += hit(ranked[:5], {}, ())

    total = len(questions)
    result: dict[str, float] = {
        "coverage": round(100 * covered_count / total, 1),
        **{key: round(100 * value / total, 1) for key, value in counts.items()},
        "rerank@5": round(100 * reranked / max(covered_count, 1), 1),
        "unlabelled in top 5": unlabelled,
    }

    return result


def main() -> None:
    """Print a table per split and save it to data/deep_answered_summary.json."""
    questions = load_questions()
    labels = load_labels()
    found = load_deep_found()
    original = all_orderings()
    rows: dict[str, Ranker] = {name: original[name] for name in BASELINES if name in original}
    rows |= deep_orderings()

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for split in SPLITS:
        part = [question for question in questions if split == "all" or question["split"] == split]
        summary[split] = {
            name: metrics(part, labels, name, ordering, found) for name, ordering in rows.items()
        }
        print(f"\n== {split} ({len(part)} questions)")
        for name, values in summary[split].items():
            print(f"  {name:28s} {values}")

    (DATA / "deep_answered_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
