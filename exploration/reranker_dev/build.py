"""Build the dev reranking set: candidates of today's first search, scored by today's reranker.

The set lets other rerankers (smaller models, ColBERT, shorter inputs) be compared with the
current one offline, on exactly the inputs it sees in production:

- **Questions:** every semantic question of the `dev` (180) and `dev_slack` (200) splits of
  `exploration/training_questions/data/questions.jsonl`, each with one right document. The
  `train` split is left out: the fine-tuned gte was trained on it.
- **Candidates:** what the first search (`hybrid-ft-union`) hands the reranker: the fine-tuned
  gte's top 50 and BM25's top 50, the union of both (about 97 documents), with each
  candidate's gte rank, BM25 rank, fusion score and whether it is the right document.
- **Scores:** gte-reranker-modernbert-base on `title + text`, at 1,024 tokens (today's setting)
  and at 8,192 (the whole text), with the seconds each question took.

Output (in `exploration/reranker_dev/data/`):
    candidates.jsonl   one line per question, resumable (written as each question finishes)
    documents.parquet  doc_id, title, text: every candidate's input, stored once
    summary.json       recall@1/5/10/20 of the fused order and of each reranker setting

Run from the main checkout (it has the data and the models), with the progress page open:
    uv run python <workspace>/exploration/reranker_dev/build.py [--workflow KEY]
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import cast

import pandas as pd
from orderings import (
    Candidate,
    DevQuestion,
    FoundCandidate,
    Ordering,
    Question,
    SourceQuestion,
    all_orderings,
)

from train_rag.database import get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.pipelines.retrieval import CANDIDATES, RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress
from train_rag.reranker import FULL_TOKENS, SHORT_TOKENS, GteModernBertReranker

QUESTIONS = Path("exploration/training_questions/data/questions.jsonl")
DATA = Path(__file__).parent / "data"
SPLITS = ("dev", "dev_slack")
REPORTED = ("fused order", "cross-encoder 1,024", "cross-encoder 8,192")


def load_questions() -> list[DevQuestion]:
    """Load the valid semantic questions of the dev splits, each with a stable id.

    Returns:
        The questions with an added `qid` ("<split>-NNN").
    """
    lines = QUESTIONS.read_text().splitlines()
    rows = [cast(SourceQuestion, json.loads(line)) for line in lines if line.strip()]
    picked = [
        row
        for row in rows
        if row["question_type"] == "semantic" and row["split"] in SPLITS and row["valid"]
    ]

    return [
        DevQuestion(**row, qid=f"{row['split']}-{index:03d}") for index, row in enumerate(picked)
    ]


async def collect(
    questions: list[DevQuestion], out: Path, workflow: str | None
) -> dict[str, list[FoundCandidate]]:
    """Run the first search without a reranker; keep each question's candidates and texts.

    Args:
        questions: Dev questions from load_questions.
        out: Output folder; documents.parquet is written there.
        workflow: Progress-page workflow key, or None.

    Returns:
        Candidates per question id, in fused order.
    """
    pipeline = RetrievalPipeline.load(rerank=False, embedder=FineTunedGteEmbedder())
    found: dict[str, list[FoundCandidate]] = {}
    texts: dict[str, tuple[str, str]] = {}
    try:
        with RunProgress(
            get_engine(),
            "dev reranking set: first-search candidates",
            len(questions),
            unit="questions",
            workflow=workflow,
            step="Collect first-search candidates",
        ) as progress:
            for question in questions:
                docs = await pipeline.run(question["question"], limit=2 * CANDIDATES)
                found[question["qid"]] = [
                    FoundCandidate(
                        doc_id=doc.doc_id,
                        source_type=str(doc.source_type),
                        rrf=doc.score,
                        gte_rank=doc.ranks.get("gte"),
                        bm25_rank=doc.ranks.get("bm25"),
                        is_expected=doc.doc_id == question["doc_id"],
                    )
                    for doc in docs
                ]
                for doc in docs:
                    texts.setdefault(doc.doc_id, (doc.title, doc.text))
                progress.advance("questions", 1)
    finally:
        await pipeline.close()

    out.mkdir(parents=True, exist_ok=True)
    table = [
        {"doc_id": doc_id, "title": title, "text": text} for doc_id, (title, text) in texts.items()
    ]
    pd.DataFrame(table).to_parquet(out / "documents.parquet", index=False)

    return found


def score(
    questions: list[DevQuestion],
    found: dict[str, list[FoundCandidate]],
    out: Path,
    workflow: str | None,
) -> None:
    """Score every candidate with the current reranker at both lengths; append per question.

    Args:
        questions: Dev questions from load_questions.
        found: Candidates per question id, from collect.
        out: Output folder with documents.parquet; candidates.jsonl is appended there.
        workflow: Progress-page workflow key, or None.
    """
    docs = pd.read_parquet(out / "documents.parquet").set_index("doc_id")
    out_file = out / "candidates.jsonl"
    done = (
        {json.loads(line)["qid"] for line in out_file.read_text().splitlines()}
        if out_file.exists()
        else set()
    )
    todo = [question for question in questions if question["qid"] not in done]

    short = GteModernBertReranker(max_tokens=SHORT_TOKENS)
    full = GteModernBertReranker(max_tokens=FULL_TOKENS)

    with (
        RunProgress(
            get_engine(),
            "dev reranking set: score with the current reranker",
            len(todo),
            unit="questions",
            workflow=workflow,
            step="Score with the current reranker (1,024 and 8,192 tokens)",
        ) as progress,
        out_file.open("a") as handle,
    ):
        for question in todo:
            cands = found[question["qid"]]
            inputs = [
                f"{docs.at[cand['doc_id'], 'title']}\n{docs.at[cand['doc_id'], 'text']}"
                for cand in cands
            ]
            clean = normalize_text(question["question"])
            started = time.perf_counter()
            scores_short = short.score(clean, inputs)
            seconds_short = time.perf_counter() - started
            started = time.perf_counter()
            scores_full = full.score(clean, inputs)
            seconds_full = time.perf_counter() - started

            scored = [
                Candidate(**cand, rerank_1024=value_short, rerank_8192=value_full)
                for cand, value_short, value_full in zip(
                    cands, scores_short, scores_full, strict=True
                )
            ]
            record = Question(
                qid=question["qid"],
                split=question["split"],
                source_type=question["source_type"],
                question=question["question"],
                expected_doc_id=question["doc_id"],
                expected_in_candidates=any(cand["is_expected"] for cand in scored),
                seconds={"rerank_1024": seconds_short, "rerank_8192": seconds_full},
                candidates=scored,
            )
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            progress.advance(
                "questions", 1, note=f"last: {seconds_short:.1f} s / {seconds_full:.1f} s"
            )


def recall(questions: list[Question], ordering: Ordering) -> dict[str, float]:
    """Compute recall@1/5/10/20 of the source document under an ordering.

    Args:
        questions: Scored questions.
        ordering: How to rank each question's candidates.

    Returns:
        Recall at each depth, in percent.
    """
    result: dict[str, float] = {}
    for depth in (1, 5, 10, 20):
        hits = sum(
            any(cand["is_expected"] for cand in ordering(question)[:depth])
            for question in questions
        )
        result[f"recall@{depth}"] = round(100 * hits / len(questions), 1)

    return result


def summarise(out: Path) -> None:
    """Write recall per ordering and split, plus timing, to summary.json.

    Args:
        out: Output folder with candidates.jsonl.
    """
    lines = (out / "candidates.jsonl").read_text().splitlines()
    questions = [cast(Question, json.loads(line)) for line in lines]
    orderings = all_orderings()

    splits: dict[str, dict[str, object]] = {}
    for split in ("all", *SPLITS):
        part = [question for question in questions if split == "all" or question["split"] == split]
        if not part:
            continue
        in_candidates = sum(question["expected_in_candidates"] for question in part)
        splits[split] = {
            "questions": len(part),
            "expected_in_candidates": round(100 * in_candidates / len(part), 1),
            "candidates_avg": round(
                sum(len(question["candidates"]) for question in part) / len(part), 1
            ),
            **{name: recall(part, orderings[name]) for name in REPORTED},
            "seconds_per_question": {
                key: round(sum(question["seconds"][key] for question in part) / len(part), 2)
                for key in ("rerank_1024", "rerank_8192")
            },
        }

    summary = {"questions": len(questions), "splits": splits}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    """Collect candidates (once), score them (resumable), summarise."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", help="progress-page workflow key")
    parser.add_argument(
        "--limit", type=int, help="smoke test: first N questions, written to data-smoke/"
    )
    args = parser.parse_args()

    questions = load_questions()
    out = DATA
    if args.limit:
        out = DATA.parent / "data-smoke"
        questions = questions[: args.limit]

    found_file = out / "found.json"
    if found_file.exists():
        found = cast(dict[str, list[FoundCandidate]], json.loads(found_file.read_text()))
    else:
        found = asyncio.run(collect(questions, out, args.workflow))
        found_file.write_text(json.dumps(found))

    score(questions, found, out, args.workflow)
    summarise(out)


if __name__ == "__main__":
    main()
