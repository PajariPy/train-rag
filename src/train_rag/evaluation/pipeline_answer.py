"""A RAG system that answers from the documents the retrieval pipeline finds."""

import re
from typing import ClassVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from train_rag.evaluation.system import NOT_FOUND_ANSWER, RagSystem
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.evaluation import SystemAnswer
from train_rag.models.retrieval import SearchHit
from train_rag.prompts import prompt
from train_rag.retrieval import Retriever

GEMINI_MODEL = "gemini-3.8-flash"
# A document's text is cut here (about 3,000 tokens), so one long transcript can't
# dominate the cost of an answer. 95% of documents are shorter.
MAX_DOCUMENT_CHARS = 12_000
ANSWER_INSTRUCTIONS = prompt("document_answer_instructions", not_found=NOT_FOUND_ANSWER)
# An inline citation as the model writes it: document numbers in brackets, "[2]" or "[2, 4]".
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class CitedAnswer(BaseModel):
    """The model's answer and the numbers of the documents it is taken from."""

    answer: str
    cited_documents: list[int]


def cited_doc_ids(numbers: list[int], hits: list[SearchHit]) -> list[str]:
    """Turn the document numbers a model cites into document ids.

    Args:
        numbers: Cited numbers, 1 for the first document; ones out of range are dropped.
        hits: The documents as numbered in the prompt.

    Returns:
        The cited documents' ids, without repeats, in the order cited.
    """
    return list(
        dict.fromkeys(hits[number - 1].doc_id for number in numbers if 1 <= number <= len(hits))
    )


def link_citations(answer: str, hits: list[SearchHit]) -> str:
    """Replace the document numbers cited in an answer with the documents' ids.

    The model cites the numbers it was shown ("[2]", "[2, 4]"); the app needs ids to
    open the document on click, so each becomes "[doc_id]". Numbers out of range are
    dropped, and a citation left empty is removed.

    Args:
        answer: The model's answer with inline citations.
        hits: The documents as numbered in the prompt.

    Returns:
        The answer with "[doc_id]" citations.
    """

    def replace(match: re.Match[str]) -> str:
        numbers = [int(n) for n in match.group(1).split(",")]
        return "".join(f"[{hits[n - 1].doc_id}]" for n in numbers if 1 <= n <= len(hits))

    return CITATION.sub(replace, answer)


def inline_doc_ids(answer: str) -> list[str]:
    """The document ids cited inline in an answer, in order of first citation.

    Args:
        answer: An answer whose citations `link_citations` turned into ids.

    Returns:
        The cited ids, without repeats.
    """
    return list(dict.fromkeys(re.findall(r"\[(dsid_[0-9a-f]+)\]", answer)))


def documents_context(hits: list[SearchHit]) -> str:
    """Lay out the retrieved documents for the prompt.

    Args:
        hits: Retrieved documents, best first; their text starts with the title.

    Returns:
        The numbered documents, each cut to `MAX_DOCUMENT_CHARS`.
    """
    return "\n\n".join(
        f"[{index}] {hit.source_type} ({hit.doc_id}):\n{hit.text[:MAX_DOCUMENT_CHARS]}"
        for index, hit in enumerate(hits, 1)
    )


def answer_from_hits(
    client: genai.Client, usage: UsageMeter, question: str, hits: list[SearchHit], note: str = ""
) -> SystemAnswer:
    """Let Gemini answer from documents, citing them inline, and link the citations.

    Args:
        client: Gemini client.
        usage: Counts the answer's tokens.
        question: The question text.
        hits: The documents, best first, each cut to `MAX_DOCUMENT_CHARS`.
        note: An extra instruction for this answer, added after the documents.

    Returns:
        The answer with "[doc_id]" citations, the documents, and the ones it cites.
    """
    contents = f"Question: {question}\n\nDocuments:\n\n{documents_context(hits)}"
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{contents}\n\n{note}" if note else contents,
        config=types.GenerateContentConfig(
            system_instruction=ANSWER_INSTRUCTIONS,
            response_mime_type="application/json",
            response_schema=CitedAnswer,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        ),
    )
    usage.add(response)
    parsed = response.parsed
    empty = CitedAnswer(answer="", cited_documents=[])
    cited = parsed if isinstance(parsed, CitedAnswer) else empty
    answer = link_citations(cited.answer.strip(), hits)

    return SystemAnswer(
        answer=answer or NOT_FOUND_ANSWER,
        retrieved_doc_ids=list(dict.fromkeys(hit.doc_id for hit in hits)),
        # The listed sources first (main one first), then any cited only inline.
        cited_doc_ids=list(
            dict.fromkeys(cited_doc_ids(cited.cited_documents, hits) + inline_doc_ids(answer))
        ),
    )


class PipelineAnswerSystem(RagSystem):
    """Retrieve with BM25 and gte, fused and reranked, then let Gemini answer."""

    name: ClassVar[str] = "hybrid-rerank"

    def __init__(
        self, retriever: Retriever, client: genai.Client, usage: UsageMeter, top_k: int = 5
    ) -> None:
        """Set up the system.

        Args:
            retriever: The retrieval pipeline.
            client: Gemini client for the answer.
            usage: Counts the tokens of every answer.
            top_k: Documents shown to the model.
        """
        self.retriever = retriever
        self.client = client
        self.usage = usage
        self.top_k = top_k

    def answer(self, question: str) -> SystemAnswer:
        """Retrieve documents and answer from them.

        Args:
            question: The question text.

        Returns:
            The answer, the retrieved documents, best first, and the ones it cites.
        """
        hits = self.retriever.retrieve(question, self.top_k)

        return answer_from_hits(self.client, self.usage, question, hits)


class FineTunedPipelineAnswerSystem(PipelineAnswerSystem):
    """The same pipeline with the fine-tuned gte, so its runs are stored under their own name."""

    name: ClassVar[str] = "hybrid-ft-rerank"
