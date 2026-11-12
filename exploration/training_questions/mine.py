"""Search, rerank and pick hard negatives for every generated question.

Design: docs/brainstorm/005_synthetic_training_questions.md and 004 (the drop rule).
Plan task 3.

For each valid question in data/questions.jsonl:
1. search as production does: gte top 50 + BM25 top 50, fused with RRF to 50;
2. add the source document if search missed it, and rerank all of them (title + text);
3. apply the biggest-drop rule: skip the question if its source isn't above the drop,
   otherwise take up to 5 negatives just below the drop, never an evaluation document.

Reranker scores are cached per question in data/rerank_scores.json, so a rerun only scores
what's missing. Writes data/training_set.jsonl and data/dev_set.jsonl from the cache and
prints the report. Free, local; about 2.5 hours for all questions on the laptop.

Usage:
    uv run python exploration/training_questions/mine.py --split dev --limit 50   # pilot
    uv run python exploration/training_questions/mine.py                          # everything
"""

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from sqlalchemy import text

from train_rag.database import get_engine
from train_rag.fusion import reciprocal_rank_fusion
from train_rag.pipelines.retrieval import RERANK_DEPTH, RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress
from train_rag.reranker import GteModernBertReranker
from train_rag.training.drop_rule import (
    biggest_drop,
    candidates_with_positive,
    negatives_excluded,
    pick_negatives,
)

DATA = Path(__file__).parent / "data"
QUESTIONS = DATA / "questions.jsonl"
EXCLUDED = DATA / "excluded.json"
SCORES = DATA / "rerank_scores.json"
OUTPUT = {"train": DATA / "training_set.jsonl", "dev": DATA / "dev_set.jsonl"}
SAVE_EVERY = 25


async def fused_candidates(pipeline: RetrievalPipeline, question: str) -> list[str]:
    """Search one question as production does.

    Args:
        pipeline: The loaded pipeline (no reranker).
        question: The question text.

    Returns:
        The fused top 50 doc ids, best first.
    """
    clean = normalize_text(question)
    semantic, keyword = await asyncio.gather(
        pipeline.semantic_search(clean), pipeline.keyword_search(clean)
    )

    return [
        document.doc_id
        for document in reciprocal_rank_fusion({"gte": semantic, "bm25": keyword}, RERANK_DEPTH)
    ]


def load_texts(doc_ids: list[str]) -> dict[str, str]:
    """Title + text of each document, as the pipeline feeds the reranker.

    Args:
        doc_ids: The documents.

    Returns:
        doc_id -> the title, a line break and the text.
    """
    with get_engine().connect() as connection:
        rows = connection.execute(
            text("select doc_id, title, content from documents where doc_id = any(:ids)"),
            {"ids": doc_ids},
        ).all()

    return {str(row.doc_id): f"{row.title}\n{row.content}" for row in rows}


async def score_missing(questions: list[dict[str, Any]], scores: dict[str, Any]) -> None:
    """Search and rerank every question not in the cache, saving as it goes.

    Args:
        questions: The questions to score.
        scores: The cache: doc_id of the source -> {candidate doc_id: reranker score}.
    """
    todo = [question for question in questions if question["doc_id"] not in scores]
    if not todo:
        return
    pipeline = RetrievalPipeline.load(rerank=False)
    reranker = GteModernBertReranker()
    with RunProgress(
        get_engine(),
        "mine hard negatives for training questions (free)",
        len(todo),
        unit="questions",
    ) as progress:
        for count, question in enumerate(todo, 1):
            fused = await fused_candidates(pipeline, question["question"])
            doc_ids = candidates_with_positive(fused, question["doc_id"])
            texts = load_texts(doc_ids)
            doc_ids = [doc_id for doc_id in doc_ids if doc_id in texts]
            values = reranker.score(
                normalize_text(question["question"]), [texts[doc_id] for doc_id in doc_ids]
            )
            scores[question["doc_id"]] = dict(zip(doc_ids, values, strict=True))
            progress.advance(question["split"], 1, note="local, $0")
            if count % SAVE_EVERY == 0:
                SCORES.write_text(json.dumps(scores))
    SCORES.write_text(json.dumps(scores))
    await pipeline.close()


def write_sets(questions: list[dict[str, Any]], scores: dict[str, Any], limited: bool) -> None:
    """Apply the drop rule, write the train and dev sets, and print the report.

    Only the splits that were loaded are written, and nothing is written for a `--limit`
    pilot, so a pilot never overwrites a finished set. Negatives never include an evaluation
    document or another split's answer.

    Args:
        questions: The scored questions.
        scores: The reranker scores per question.
        limited: Whether `--limit` cut the questions (then only the report is printed).
    """
    evaluation = set(json.loads(EXCLUDED.read_text())["evaluation"])
    every_question = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line]
    positives: dict[str, set[str]] = defaultdict(set)
    for row in every_question:
        positives[row["split"]].add(row["doc_id"])
    excluded = {split: negatives_excluded(split, evaluation, positives) for split in positives}
    rows: dict[str, list[str]] = defaultdict(list)
    kept: Counter[tuple[str, str]] = Counter()
    total: Counter[tuple[str, str]] = Counter()
    grey: dict[str, list[int]] = defaultdict(list)
    negatives_per_question: dict[str, list[int]] = defaultdict(list)
    for question in questions:
        question_scores = scores[question["doc_id"]]
        key = (question["split"], question["question_type"])
        total[key] += 1
        ordered = sorted(question_scores.values(), reverse=True)
        grey[question["split"]].append(biggest_drop(ordered))
        negatives = pick_negatives(question_scores, question["doc_id"], excluded[question["split"]])
        if negatives is None:
            continue
        kept[key] += 1
        negatives_per_question[question["split"]].append(len(negatives))
        rows[question["split"]].append(
            json.dumps(
                {
                    "id": question["doc_id"],
                    "question_type": question["question_type"],
                    "question": question["question"],
                    "positive": question["doc_id"],
                    "negatives": negatives,
                    "scores": question_scores,
                }
            )
        )
    if limited:
        print("--limit given: report only, the sets on disk are left as they are")
    else:
        for split in grey:
            OUTPUT[split].write_text("".join(row + "\n" for row in rows[split]))
    for split in ("train", "dev"):
        if not grey[split]:
            continue
        print(f"\n{split}: kept {len(rows[split])} of {len(grey[split])} questions")
        for (row_split, question_type), count in sorted(total.items()):
            if row_split == split:
                kept_count = kept[(split, question_type)]
                print(
                    f"  {question_type:25s} kept {kept_count:4d} of {count:4d} "
                    f"(skipped {1 - kept_count / count:.0%})"
                )
        print(
            f"  above the drop: median {median(grey[split])}, "
            f"largest {max(grey[split])}; negatives per question: median "
            f"{median(negatives_per_question[split]) if negatives_per_question[split] else 0}"
        )


def main() -> None:
    """Score what's missing, then write the sets and the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    questions = [
        row
        for row in (json.loads(line) for line in QUESTIONS.read_text().splitlines() if line)
        if row["valid"] and (args.split is None or row["split"] == args.split)
    ][: args.limit]
    scores: dict[str, Any] = json.loads(SCORES.read_text()) if SCORES.exists() else {}
    asyncio.run(score_missing(questions, scores))
    write_sets(
        [question for question in questions if question["doc_id"] in scores],
        scores,
        limited=args.limit is not None,
    )


if __name__ == "__main__":
    main()
