"""Ask Gemini what each shown document, on its own, answers to its sampled question.

For every question in sample.json, each of its gte top 2 documents and its expected document
is given to Gemini alone with the question. Gemini explains its reading, says how fully the
document answers it (full, partial or none), and gives the answer with the passage it rests
on, quoted word for word. The compare page shows these next to the gold answer and
highlights the quotes, so labelling ambiguity means comparing short answers instead of
reading whole documents.

Usage: uv run python exploration/question_ambiguity/answers.py
"""

import json
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
from llm_cache import JsonCache, hash_key
from pydantic import BaseModel
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.evaluation.judge import GEMINI_MODEL
from train_rag.evaluation.usage import UsageMeter
from train_rag.processing.normalize import normalize_content
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.repository.question import load_questions
from train_rag.tables import DocumentRecord

HERE = Path(__file__).parent
SAMPLE = HERE / "sample.json"
OUT = HERE / "doc_answers.json"
PROMPTS = HERE / "prompts.yaml"
# Every rating ever made, keyed by a hash of the model, the instructions and the full
# prompt: rerunning never pays twice, and a changed prompt or document gets a new rating.
CACHE = HERE / "rating_cache.json"
WORKERS = 8


class DocumentAnswer(BaseModel):
    """What one document, on its own, says in answer to a question, and why."""

    reasoning: str
    coverage: Literal["full", "partial", "none"]
    answer: str
    quote: str


@cache
def prompts() -> dict[str, str]:
    """The prompts in prompts.yaml."""
    loaded: dict[str, str] = yaml.safe_load(PROMPTS.read_text())

    return loaded


def prompt_text(question: str, document: DocumentRecord) -> str:
    """The message for one question and one document."""
    return prompts()["document_answer_prompt"].format(
        question=question,
        source=document.source_type,
        name=document.title.splitlines()[0] if document.title else "",
        text=normalize_content(document.content),
    )


def cache_key(question: str, document: DocumentRecord) -> str:
    """A hash of everything that decides a rating: model, instructions and prompt."""
    return hash_key(
        [GEMINI_MODEL, prompts()["document_answer_instructions"], prompt_text(question, document)]
    )


class RatingCache(JsonCache):
    """Ratings stored in CACHE, shared by every script and notebook that rates documents."""

    def __init__(self, path: Path = CACHE) -> None:
        """Load the ratings saved so far.

        Args:
            path: The cache file.
        """
        super().__init__(path)

    def rating(self, question: str, document: DocumentRecord) -> DocumentAnswer | None:
        """The saved rating of one document for one question, or None."""
        entry = self.get(cache_key(question, document))

        return DocumentAnswer.model_validate(entry) if entry is not None else None

    def remember(self, question: str, document: DocumentRecord, answer: DocumentAnswer) -> None:
        """Remember a rating; `save` writes it to disk."""
        self.put(cache_key(question, document), answer.model_dump())


def ask(
    client: genai.Client, usage: UsageMeter, question: str, document: DocumentRecord
) -> DocumentAnswer:
    """Ask Gemini what one document answers to the question."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt_text(question, document),
        config=types.GenerateContentConfig(
            system_instruction=prompts()["document_answer_instructions"],
            response_mime_type="application/json",
            response_schema=DocumentAnswer,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        ),
    )
    usage.add(response)
    if not isinstance(response.parsed, DocumentAnswer):
        raise ValueError(f"No answer for {document.doc_id}")

    return response.parsed


def rate(
    client: genai.Client | None,
    usage: UsageMeter,
    cache: RatingCache,
    question: str,
    document: DocumentRecord,
) -> DocumentAnswer | None:
    """The cached rating, or a new one from Gemini when a client is given.

    Args:
        client: Gemini client, or None to only read the cache.
        usage: Counts the tokens of new ratings.
        cache: Saved ratings; a new rating is added to it.
        question: The question.
        document: The document to rate.

    Returns:
        The rating, or None when it isn't cached and no client was given.
    """
    cached = cache.rating(question, document)
    if cached is not None or client is None:
        return cached
    answer = ask(client, usage, question, document)
    cache.remember(question, document, answer)

    return answer


def main() -> None:
    """Answer every sampled question from each of its shown documents."""
    load_dotenv(".env")
    sample = json.loads(SAMPLE.read_text())
    with Session(get_engine()) as session:
        questions = {
            question.id: question
            for question in load_questions(session, EnterpriseRagBenchLoader.name)
        }
        pairs = list(
            dict.fromkeys(
                (row["question_id"], doc_id)
                for row in sample
                for doc_id in [
                    *(hit["doc_id"] for hit in row["top2"]),
                    *questions[row["question_id"]].expected_doc_ids,
                ]
            )
        )
        ids = ids_of_doc_ids(
            session, EnterpriseRagBenchLoader.name, sorted({doc_id for _, doc_id in pairs})
        )
        documents = {document.doc_id: document for document in load_documents(session, sorted(ids))}

    client = genai.Client()
    usage = UsageMeter()
    cache = RatingCache()
    try:
        with ThreadPoolExecutor(WORKERS) as pool:
            answers = list(
                pool.map(
                    lambda pair: rate(
                        client, usage, cache, questions[pair[0]].text, documents[pair[1]]
                    ),
                    pairs,
                )
            )
    finally:
        cache.save()
    results: dict[str, dict[str, dict[str, object]]] = {}
    for (question_id, doc_id), answer in zip(pairs, answers, strict=True):
        assert answer is not None
        results.setdefault(question_id, {})[doc_id] = answer.model_dump()
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"{len(pairs)} answers written to {OUT}")
    print(usage.report())


if __name__ == "__main__":
    main()
