"""Ask Gemini which of the documents a system was given its answer came from.

For answers recorded before the pipeline cited its documents (run 8 and older): Gemini
gets the question, the recorded answer and the documents the system was given, numbered
as the system saw them, and names the ones the answer was taken from. Results are cached
in citation_cache.json, like the document ratings.
"""

from pathlib import Path

from answers import prompts
from google import genai
from google.genai import types
from llm_cache import JsonCache, hash_key
from pydantic import BaseModel

from train_rag.evaluation.judge import GEMINI_MODEL
from train_rag.evaluation.pipeline_answer import MAX_DOCUMENT_CHARS
from train_rag.evaluation.usage import UsageMeter
from train_rag.tables import DocumentRecord

HERE = Path(__file__).parent
CITATION_CACHE = HERE / "citation_cache.json"


class Citation(BaseModel):
    """Which numbered documents an answer was taken from, and why."""

    reasoning: str
    cited_documents: list[int]


def prompt_text(question: str, answer: str, documents: list[DocumentRecord]) -> str:
    """The message for one answer, with the documents numbered as the system saw them."""
    numbered = "\n\n".join(
        f"[{index}] {document.source_type} ({document.doc_id}):\n"
        + f"{document.title}\n{document.content}"[:MAX_DOCUMENT_CHARS]
        for index, document in enumerate(documents, 1)
    )

    return prompts()["citation_prompt"].format(question=question, answer=answer, documents=numbered)


def cache_key(question: str, answer: str, documents: list[DocumentRecord]) -> str:
    """A hash of everything that decides a citation: model, instructions and prompt."""
    return hash_key(
        [
            GEMINI_MODEL,
            prompts()["citation_instructions"],
            prompt_text(question, answer, documents),
        ]
    )


def cited_ids(citation: Citation, documents: list[DocumentRecord]) -> list[str]:
    """The cited documents' ids, in the order cited; numbers out of range are dropped."""
    return list(
        dict.fromkeys(
            documents[number - 1].doc_id
            for number in citation.cited_documents
            if 1 <= number <= len(documents)
        )
    )


def cite(
    client: genai.Client | None,
    usage: UsageMeter,
    citation_cache: JsonCache,
    question: str,
    answer: str,
    documents: list[DocumentRecord],
) -> Citation | None:
    """The cached citation, or a new one from Gemini when a client is given.

    Args:
        client: Gemini client, or None to only read the cache.
        usage: Counts the tokens of new citations.
        citation_cache: Saved citations; a new one is added to it.
        question: The question.
        answer: The system's answer.
        documents: The documents the system was given, in its order.

    Returns:
        The citation, or None when it isn't cached and no client was given.
    """
    key = cache_key(question, answer, documents)
    entry = citation_cache.get(key)
    if entry is not None:
        return Citation.model_validate(entry)
    if client is None:
        return None
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt_text(question, answer, documents),
        config=types.GenerateContentConfig(
            system_instruction=prompts()["citation_instructions"],
            response_mime_type="application/json",
            response_schema=Citation,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        ),
    )
    usage.add(response)
    if not isinstance(response.parsed, Citation):
        raise ValueError(f"No citation for: {question}")
    citation_cache.put(key, response.parsed.model_dump())

    return response.parsed
