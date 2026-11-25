"""Collect deeper first-search candidates for the dev reranking set: gte and BM25 top 200 each.

Today the reranker sees gte's top 50 and BM25's top 50 (~97 candidates), and the right document
is among them for only about half the questions. ColBERT is cheap enough to score hundreds of
candidates, so this collects each search's top 200 (~400 candidates per question), keeping each
candidate's gte and BM25 rank. From those ranks, `report` shows how often the right document is
among the candidates at depth 50, 100, 150 and 200 per search.

The pipeline's depth is a module constant (`CANDIDATES`, 50); this exploration script raises it
for its own process only, so production code stays unchanged.

Output in `data/`:
    deep_found.json         {qid: [candidate, ...]} with gte_rank / bm25_rank / is_expected
    deep_documents.parquet  doc_id, title, text of every candidate not in documents.parquet
    deep_summary.json       right document among the candidates, per depth and split

Run from the main checkout:
    uv run python <workspace>/exploration/reranker_dev/deep_candidates.py [--workflow KEY]
"""

import argparse
import asyncio
import json
from typing import cast

import pandas as pd
from orderings import DATA, FoundCandidate, Question, load_questions, within

import train_rag.pipelines.retrieval as retrieval
from train_rag.database import get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.pipelines.retrieval import RetrievalPipeline
from train_rag.progress.recorder import RunProgress

DEPTH = 200
REPORT_DEPTHS = (50, 100, 150, 200)
SPLITS = ("all", "dev", "dev_slack")


async def collect(
    questions: list[Question], workflow: str | None
) -> tuple[dict[str, list[FoundCandidate]], dict[str, tuple[str, str]]]:
    """Run the first search at depth 200 per search, without a reranker.

    Args:
        questions: The dev reranking set (its expected documents mark the right candidate).
        workflow: Progress-page workflow key, or None.

    Returns:
        Candidates per question id in fused order, and title and text per document id.
    """
    retrieval.CANDIDATES = DEPTH  # this process only: deeper gte and BM25 lists
    pipeline = RetrievalPipeline.load(rerank=False, embedder=FineTunedGteEmbedder())
    found: dict[str, list[FoundCandidate]] = {}
    texts: dict[str, tuple[str, str]] = {}

    try:
        with RunProgress(
            get_engine(),
            "dev reranking set: deep candidates (gte + BM25 top 200)",
            len(questions),
            unit="questions",
            workflow=workflow,
            step="Collect deeper candidates (top 200 each)",
        ) as progress:
            for question in questions:
                docs = await pipeline.run(question["question"], limit=2 * DEPTH)
                found[question["qid"]] = [
                    FoundCandidate(
                        doc_id=doc.doc_id,
                        source_type=str(doc.source_type),
                        rrf=doc.score,
                        gte_rank=doc.ranks.get("gte"),
                        bm25_rank=doc.ranks.get("bm25"),
                        is_expected=doc.doc_id == question["expected_doc_id"],
                    )
                    for doc in docs
                ]
                for doc in docs:
                    texts.setdefault(doc.doc_id, (doc.title, doc.text))
                progress.advance("questions", 1)
    finally:
        await pipeline.close()

    return found, texts


def report(questions: list[Question], found: dict[str, list[FoundCandidate]]) -> dict[str, object]:
    """Measure, per depth and split, how often the right document is among the candidates.

    Args:
        questions: The dev reranking set.
        found: Deep candidates per question id.

    Returns:
        Per split: questions, and per depth the share with the right document and the
        average number of candidates.
    """
    summary: dict[str, object] = {}
    for split in SPLITS:
        part = [question for question in questions if split == "all" or question["split"] == split]
        depths: dict[str, dict[str, float]] = {}
        for depth in REPORT_DEPTHS:
            kept = [
                [cand for cand in found[question["qid"]] if within(cand, depth)]
                for question in part
            ]
            hits = sum(any(cand["is_expected"] for cand in cands) for cands in kept)
            depths[f"top {depth} each"] = {
                "right_document_in_candidates": round(100 * hits / len(part), 1),
                "candidates_avg": round(sum(len(cands) for cands in kept) / len(part), 1),
            }
        summary[split] = {"questions": len(part), **depths}

    return summary


def main() -> None:
    """Collect (once), save the new texts, print and save the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", help="progress-page workflow key")
    args = parser.parse_args()

    questions = load_questions()
    found_file = DATA / "deep_found.json"
    if found_file.exists():
        found = cast(dict[str, list[FoundCandidate]], json.loads(found_file.read_text()))
    else:
        found, texts = asyncio.run(collect(questions, args.workflow))
        found_file.write_text(json.dumps(found))
        known = set(pd.read_parquet(DATA / "documents.parquet", columns=["doc_id"])["doc_id"])
        rows = [
            {"doc_id": doc_id, "title": title, "text": text}
            for doc_id, (title, text) in texts.items()
            if doc_id not in known
        ]
        pd.DataFrame(rows).to_parquet(DATA / "deep_documents.parquet", index=False)
        print(f"{len(rows)} new documents saved")

    summary = report(questions, found)
    (DATA / "deep_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
