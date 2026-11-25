"""Records and orderings of the dev reranking set, shared by every script in this folder.

An ordering puts a question's candidates in rank order, best first:

- `fused order`: the first search's own order (no reranker).
- `cross-encoder 1,024` / `cross-encoder 8,192`: today's reranker at both lengths.
- `colbert <name>`: MaxSim scores of a ColBERT model (`data/scores_<name>.json`, colbert.py).
- `colbert <name> top N + cross-encoder`: the ColBERT model's top N, reordered by the
  cross-encoder (1,024 tokens); the cross-encoder then reads N documents instead of ~97.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Literal, TypedDict, cast

DATA = Path(__file__).parent / "data"
TWO_STAGE_DEPTHS = (10, 20, 30)
DEEP_DEPTHS = (50, 100, 150, 200)

ScoreField = Literal["rerank_1024", "rerank_8192"]
CROSS_ENCODER: Final[ScoreField] = "rerank_1024"


class SourceQuestion(TypedDict):
    """A generated question (`exploration/training_questions/data/questions.jsonl`)."""

    doc_id: str
    source_type: str
    question_type: str
    split: str
    question: str
    valid: bool


class DevQuestion(SourceQuestion):
    """A generated question picked for the dev reranking set, with a stable id."""

    qid: str


class FoundCandidate(TypedDict):
    """A candidate the first search found for a question, before reranking."""

    doc_id: str
    source_type: str
    rrf: float
    gte_rank: int | None
    bm25_rank: int | None
    is_expected: bool


class Candidate(FoundCandidate):
    """A candidate with the cross-encoder's scores at both lengths."""

    rerank_1024: float
    rerank_8192: float


class Question(TypedDict):
    """One question of the dev reranking set with its scored candidates."""

    qid: str
    split: str
    source_type: str
    question: str
    expected_doc_id: str
    expected_in_candidates: bool
    seconds: dict[str, float]
    candidates: list[Candidate]


Ordering = Callable[[Question], list[Candidate]]
Ranker = Callable[[Question], Sequence[FoundCandidate]]
ScoreTable = dict[str, dict[str, float]]


def load_questions() -> list[Question]:
    """Read the dev reranking set.

    Returns:
        Every question with its candidates and the cross-encoder's scores.
    """
    lines = (DATA / "candidates.jsonl").read_text().splitlines()

    return [cast(Question, json.loads(line)) for line in lines]


def load_labels() -> dict[str, dict[str, str]]:
    """Read the Gemini answers-it labels.

    Returns:
        For each question id, each rated document id with "full", "partial" or "none";
        empty when nothing is labelled yet.
    """
    path = DATA / "labels.json"
    if not path.exists():
        return {}

    return cast(dict[str, dict[str, str]], json.loads(path.read_text()))


def load_colbert_scores() -> dict[str, ScoreTable]:
    """Read every ColBERT score file written by colbert.py.

    Returns:
        For each model name, its MaxSim score per question id and document id.
    """
    tables: dict[str, ScoreTable] = {}
    for path in sorted(DATA.glob("scores_*.json")):
        name = path.stem.removeprefix("scores_")
        if name.endswith(("-smoke", ".partial")):
            continue
        tables[name] = cast(ScoreTable, json.loads(path.read_text()))

    return tables


def by_stored_score(key: ScoreField) -> Ordering:
    """Order candidates by a score stored on them (the cross-encoder's).

    Args:
        key: The candidate field, `rerank_1024` or `rerank_8192`.

    Returns:
        The ordering.
    """

    def order(question: Question) -> list[Candidate]:
        """Sort the question's candidates by the stored score, best first.

        Args:
            question: The question to order.

        Returns:
            Its candidates in rank order.
        """
        return sorted(question["candidates"], key=lambda candidate: -candidate[key])

    return order


def by_colbert(table: ScoreTable) -> Ordering:
    """Order candidates by a ColBERT model's MaxSim scores.

    Args:
        table: The model's score per question id and document id.

    Returns:
        The ordering.
    """

    def order(question: Question) -> list[Candidate]:
        """Sort the question's candidates by MaxSim score, best first.

        Args:
            question: The question to order.

        Returns:
            Its candidates in rank order.
        """
        scores = table[question["qid"]]

        return sorted(question["candidates"], key=lambda candidate: -scores[candidate["doc_id"]])

    return order


def colbert_then_cross_encoder(table: ScoreTable, depth: int) -> Ordering:
    """Keep a ColBERT model's top `depth` candidates, then order them by the cross-encoder.

    Args:
        table: The model's score per question id and document id.
        depth: How many ColBERT candidates the cross-encoder reads.

    Returns:
        The ordering: the reordered top `depth`, then the rest in ColBERT order.
    """
    colbert = by_colbert(table)

    def order(question: Question) -> list[Candidate]:
        """Reorder the ColBERT top `depth` by the cross-encoder; keep the rest after it.

        Args:
            question: The question to order.

        Returns:
            Its candidates in rank order.
        """
        ranked = colbert(question)
        head = sorted(ranked[:depth], key=lambda candidate: -candidate[CROSS_ENCODER])

        return head + ranked[depth:]

    return order


def all_orderings() -> dict[str, Ordering]:
    """Build every ordering there are scores for.

    Returns:
        Ordering name to ordering, in report order.
    """
    orderings: dict[str, Ordering] = {
        "fused order": lambda question: question["candidates"],
        "cross-encoder 1,024": by_stored_score("rerank_1024"),
        "cross-encoder 8,192": by_stored_score("rerank_8192"),
    }
    for name, table in load_colbert_scores().items():
        orderings[f"colbert {name}"] = by_colbert(table)
        for depth in TWO_STAGE_DEPTHS:
            two_stage = colbert_then_cross_encoder(table, depth)
            orderings[f"colbert {name} top {depth} + cross-encoder"] = two_stage

    return orderings


def load_deep_found() -> dict[str, list[FoundCandidate]]:
    """Read the deep candidates (gte and BM25 top 200 each) from deep_candidates.py.

    Returns:
        Candidates per question id, in fused order.
    """
    path = DATA / "deep_found.json"

    return cast(dict[str, list[FoundCandidate]], json.loads(path.read_text()))


def load_deep_scores() -> dict[str, ScoreTable]:
    """Read every ColBERT score file colbert.py wrote for the deep candidates.

    Returns:
        For each run name, its MaxSim score per question id and document id.
    """
    tables: dict[str, ScoreTable] = {}
    for path in sorted(DATA.glob("deep_scores_*.json")):
        name = path.stem.removeprefix("deep_scores_")
        if name.endswith(("-smoke", ".partial")):
            continue
        tables[name] = cast(ScoreTable, json.loads(path.read_text()))

    return tables


def within(candidate: FoundCandidate, depth: int) -> bool:
    """Check whether gte or BM25 ranked the candidate within a depth.

    Args:
        candidate: A collected candidate.
        depth: How many results each search keeps.

    Returns:
        True when either search put it in its top `depth`.
    """
    ranks = [rank for rank in (candidate["gte_rank"], candidate["bm25_rank"]) if rank is not None]

    return any(rank <= depth for rank in ranks)


def deep_by_colbert(
    found: dict[str, list[FoundCandidate]], table: ScoreTable, depth: int
) -> Ranker:
    """Order the deep candidates within a search depth by a ColBERT model's MaxSim scores.

    Args:
        found: Deep candidates per question id.
        table: The model's score per question id and document id.
        depth: How many results each search (gte, BM25) hands over.

    Returns:
        The ordering.
    """

    def order(question: Question) -> list[FoundCandidate]:
        """Keep the candidates within the depth and sort them by MaxSim, best first.

        Args:
            question: The question to order.

        Returns:
            Its deep candidates within the depth, in rank order.
        """
        scores = table[question["qid"]]
        kept = [cand for cand in found[question["qid"]] if within(cand, depth)]

        return sorted(kept, key=lambda cand: -scores[cand["doc_id"]])

    return order


def deep_orderings() -> dict[str, Ranker]:
    """Build every deep ordering there are scores for: each run at each search depth.

    Returns:
        Ordering name ("deep <run> @<depth>") to ordering.
    """
    tables = load_deep_scores()
    if not tables:
        return {}

    found = load_deep_found()

    return {
        f"deep {name} @{depth}": deep_by_colbert(found, table, depth)
        for name, table in tables.items()
        for depth in DEEP_DEPTHS
    }
