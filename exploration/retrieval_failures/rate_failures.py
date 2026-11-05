"""Rate every document shown for each wrong answer of the cited run with the document judge.

For each wrong answer of CITED_RUN: the 5 documents given to the LLM and the expected
document, each rated on its own (full, partial or none) by the same judge as stage 2 of
notebook 006 (`document_answer_instructions` in exploration/question_ambiguity/prompts.yaml).
Ratings go to the shared rating cache, so the case page and the notebook read them, and a
rerun never pays twice.

Usage: uv run python exploration/retrieval_failures/rate_failures.py [--pay]
Without --pay it only counts the ratings that are missing and estimates their cost.
"""

import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.evaluation.usage import UsageMeter
from train_rag.models import QuestionType
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.repository.evaluation import results_of_run
from train_rag.repository.question import load_questions

HERE = Path(__file__).parent
CITED_RUN = 15
# About 2,000 input and 250 output tokens per rating, at $0.75 and $3.75 per million.
COST_PER_RATING = 2_000 * 0.75 / 1e6 + 250 * 3.75 / 1e6
sys.path.insert(0, str(HERE.parent / "question_ambiguity"))
answers = importlib.import_module("answers")


def main() -> None:
    """Rate the missing documents, or only count them without --pay."""
    pay = "--pay" in sys.argv[1:]
    dataset = EnterpriseRagBenchLoader.name
    with Session(get_engine()) as session:
        questions = {q.id: q for q in load_questions(session, dataset, QuestionType.SEMANTIC)}
        wrong = [
            r
            for r in results_of_run(session, CITED_RUN)
            if r.correct is False and r.question_id in questions
        ]
        pairs = list(
            dict.fromkeys(
                (r.question_id, doc_id)
                for r in wrong
                for doc_id in [*r.retrieved_doc_ids, *questions[r.question_id].expected_doc_ids]
            )
        )
        ids = ids_of_doc_ids(session, dataset, sorted({doc_id for _, doc_id in pairs}))
        documents = {d.doc_id: d for d in load_documents(session, sorted(ids))}

    cache = answers.RatingCache()
    missing = [
        (qid, doc_id)
        for qid, doc_id in pairs
        if doc_id in documents
        and cache.rating(questions[qid].text, documents[doc_id]) is None
    ]
    print(
        f"{len(wrong)} wrong answers, {len(pairs)} documents to rate, {len(missing)} not cached "
        f"(about ${len(missing) * COST_PER_RATING:.2f})",
        flush=True,
    )
    if not pay or not missing:
        return

    load_dotenv(HERE.parent.parent / ".env")
    client = genai.Client()
    usage = UsageMeter()

    def rate(pair: tuple[str, str]) -> None:
        answers.rate(client, usage, cache, questions[pair[0]].text, documents[pair[1]])

    try:
        with ThreadPoolExecutor(answers.WORKERS) as pool:
            for done, _ in enumerate(pool.map(rate, missing), 1):
                if done % 25 == 0:
                    print(f"{done}/{len(missing)}", flush=True)
    finally:
        cache.save()
    print(usage.report(), flush=True)


main()
