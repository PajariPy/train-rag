r"""Rate the "does this document answer the question?" pairs an ordering's top 5 needs.

The same method, prompt, model and cache as `exploration/question_ambiguity/answers.py` (and
the 17 Nov Slack ambiguity study): Gemini 3.8 Flash reads one document alone with the question
and says full / partial / none. Rated: every source document and the top 5 of the chosen
orderings (orderings.py; default today's cross-encoder at both lengths). Pairs already in the
shared cache cost nothing.

`--estimate` counts the missing pairs and prices them from the exact prompts, without calling
Gemini. Otherwise the pairs are rated (stopping past --max-dollars) and every rated pair of the
set, old and new, is written to `data/labels.json`.

Run from the main checkout (it has the .env with the Gemini key); `answers` is imported from
exploration/question_ambiguity, so put that folder on PYTHONPATH (from the main checkout):
    export PYTHONPATH=exploration/question_ambiguity
    uv run python <workspace>/exploration/reranker_dev/label.py \
        --estimate --ordering "colbert lateon"
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from answers import CACHE, WORKERS, RatingCache, prompt_text, prompts, rate
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from orderings import (
    DATA,
    Question,
    Ranker,
    all_orderings,
    deep_orderings,
    load_labels,
    load_questions,
)
from sqlmodel import Session

from train_rag.database import get_engine
from train_rag.evaluation.usage import INPUT_PRICE, OUTPUT_PRICE, UsageMeter
from train_rag.progress.recorder import RunProgress
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.tables import DocumentRecord

TOP = 5
BATCH = 40
CHARS_PER_TOKEN = 4.0
THINKING_TOKENS = 10  # measured: 6–11 per rating with thinking level "low" (24 Nov)
DEFAULT_ORDERINGS = ("cross-encoder 1,024", "cross-encoder 8,192")
RETRIES = 6
RETRY_BASE_SECONDS = 5.0
RETRY_CODES = (429, 500, 503)

Pair = tuple[Question, str]


def wanted_pairs(questions: list[Question], names: list[str]) -> list[Pair]:
    """Collect every source document and every document in the chosen orderings' top 5.

    Args:
        questions: The dev reranking set.
        names: Ordering names from orderings.py (deep ones as "deep <run> @<depth>").

    Returns:
        (question, doc_id) pairs, each once.
    """
    orderings: dict[str, Ranker] = {**all_orderings(), **deep_orderings()}
    pairs: list[Pair] = []
    for question in questions:
        doc_ids = {question["expected_doc_id"]}
        for name in names:
            doc_ids |= {candidate["doc_id"] for candidate in orderings[name](question)[:TOP]}
        pairs += [(question, doc_id) for doc_id in sorted(doc_ids)]

    return pairs


def load_records(doc_ids: set[str]) -> dict[str, DocumentRecord]:
    """Load the documents the ratings read, first row per dataset id.

    Args:
        doc_ids: Dataset ids to load.

    Returns:
        Document record by dataset id.
    """
    with Session(get_engine()) as session:
        ids = ids_of_doc_ids(session, "enterprise_rag_bench", sorted(doc_ids))
        records = load_documents(session, sorted(ids))

    first: dict[str, DocumentRecord] = {}
    for record in records:
        first.setdefault(record.doc_id, record)

    return first


def estimate(todo: list[Pair], records: dict[str, DocumentRecord]) -> None:
    """Print how much rating the missing pairs would cost.

    Args:
        todo: Pairs without a cached rating.
        records: Document record by dataset id.
    """
    cached = json.loads(CACHE.read_text()).values()
    output_tokens = statistics.mean(
        sum(len(value.get(field, "")) for field in ("reasoning", "answer", "quote"))
        / CHARS_PER_TOKEN
        for value in cached
    )
    system = len(prompts()["document_answer_instructions"])
    prompt_chars = sum(
        system + len(prompt_text(question["question"], records[doc_id]))
        for question, doc_id in todo
    )
    input_tokens = prompt_chars / CHARS_PER_TOKEN
    output_cost = len(todo) * (output_tokens + THINKING_TOKENS) * OUTPUT_PRICE / 1e6
    cost = input_tokens * INPUT_PRICE / 1e6 + output_cost

    print(f"{len(todo)} pairs to rate, ~{input_tokens / 1e6:.2f}M input tokens, about ${cost:.2f}")


def rate_with_retry(
    client: genai.Client,
    usage: UsageMeter,
    cache: RatingCache,
    question: str,
    record: DocumentRecord,
) -> None:
    """Rate one pair, waiting and retrying while Gemini is overloaded or rate-limited.

    Args:
        client: Gemini client.
        usage: Counts the tokens of new ratings.
        cache: The shared rating cache; the new rating is added to it.
        question: The question.
        record: The document to rate.

    Raises:
        errors.APIError: When Gemini still fails after every retry, or fails for another
            reason (for example depleted credits).
    """
    for attempt in range(RETRIES):
        try:
            rate(client, usage, cache, question, record)

            return
        except errors.APIError as error:
            if error.code not in RETRY_CODES or attempt == RETRIES - 1:
                raise
            time.sleep(RETRY_BASE_SECONDS * 2**attempt)


def rate_all(
    todo: list[Pair],
    records: dict[str, DocumentRecord],
    cache: RatingCache,
    args: argparse.Namespace,
) -> None:
    """Rate the missing pairs with Gemini in batches, saving the cache after each batch.

    Args:
        todo: Pairs without a cached rating.
        records: Document record by dataset id.
        cache: The shared rating cache; new ratings are added to it.
        args: Command-line arguments (spend cap, progress workflow and step).
    """
    client, usage = genai.Client(), UsageMeter()
    try:
        with (
            RunProgress(
                get_engine(),
                "dev reranking set: answers-it ratings (Gemini)",
                len(todo),
                unit="pairs",
                workflow=args.workflow,
                step=args.step,
            ) as progress,
            ThreadPoolExecutor(WORKERS) as pool,
        ):
            for start in range(0, len(todo), BATCH):
                if usage.cost() > args.max_dollars:
                    print(f"stopped: spend ${usage.cost():.2f} passed ${args.max_dollars}")
                    break
                batch = todo[start : start + BATCH]
                list(
                    pool.map(
                        lambda pair: rate_with_retry(
                            client, usage, cache, pair[0]["question"], records[pair[1]]
                        ),
                        batch,
                    )
                )
                cache.save()
                progress.advance("rating", len(batch), note=f"Gemini: ${usage.cost():.2f}")
    finally:
        cache.save()
        print(usage.report())


def save_labels(pairs: list[Pair], records: dict[str, DocumentRecord], cache: RatingCache) -> None:
    """Add every cached rating of the pairs to data/labels.json.

    Args:
        pairs: The pairs this run needed.
        records: Document record by dataset id.
        cache: The shared rating cache.
    """
    labels = load_labels()
    for question, doc_id in pairs:
        rating = cache.rating(question["question"], records[doc_id]) if doc_id in records else None
        if rating is not None:
            labels.setdefault(question["qid"], {})[doc_id] = rating.coverage

    (DATA / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"labels for {len(labels)} questions written to {DATA / 'labels.json'}")


def main() -> None:
    """Find the missing pairs, then estimate or rate them, then write labels.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ordering", action="append", help="ordering name from orderings.py (repeatable)"
    )
    parser.add_argument(
        "--estimate", action="store_true", help="price the missing pairs, call nothing"
    )
    parser.add_argument("--max-dollars", type=float, default=6.0)
    parser.add_argument("--workflow")
    parser.add_argument("--step", default="Rate new top-5 pairs (Gemini)")
    args = parser.parse_args()
    load_dotenv(".env")

    questions = load_questions()
    pairs = wanted_pairs(questions, args.ordering or list(DEFAULT_ORDERINGS))
    records = load_records({doc_id for _, doc_id in pairs})
    cache = RatingCache()
    todo = [
        (question, doc_id)
        for question, doc_id in pairs
        if doc_id in records and cache.rating(question["question"], records[doc_id]) is None
    ]
    print(f"{len(pairs)} pairs, {len(pairs) - len(todo)} already rated, {len(todo)} to rate")

    if args.estimate:
        estimate(todo, records)

        return

    try:
        rate_all(todo, records, cache, args)
    finally:
        save_labels(pairs, records, cache)


if __name__ == "__main__":
    main()
