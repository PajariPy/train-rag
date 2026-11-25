"""The free first search's ranking for every semantic question, to compare with the agent's.

Runs `hybrid-ft-union` (the union of gte and BM25, reranked, no rewrites) on the 125
semantic questions and keeps each question's top 50 doc ids in `data/baseline.json`, which
the review page reads. Free (local GPU), about 15 minutes.

Usage: uv run python exploration/field_agent_review/baseline.py
"""

import json
from pathlib import Path

from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.evaluation.registry import RETRIEVERS
from train_rag.models import QuestionType
from train_rag.progress.recorder import RunProgress
from train_rag.repository.question import load_questions

OUTPUT = Path(__file__).parent / "data" / "baseline.json"
KEPT = 50


def main() -> None:
    """Rank the documents for every semantic question and save the rankings."""
    retriever = RETRIEVERS["hybrid-ft-union"](None)
    with Session(get_engine()) as session:
        questions = load_questions(
            session, EnterpriseRagBenchLoader.name, QuestionType.SEMANTIC, None
        )

    rankings: dict[str, list[str]] = {}
    title = "field agent review: first-search rankings"
    with RunProgress(get_engine(), title, len(questions), unit="questions") as progress:
        for question in questions:
            hits = retriever.retrieve(question.text, KEPT)
            rankings[question.id] = [hit.doc_id for hit in hits]
            progress.advance("ranking", 1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rankings, indent=1))
    print(f"{len(rankings)} rankings in {OUTPUT}")


if __name__ == "__main__":
    main()
