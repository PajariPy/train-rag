"""A RAG system that answers from the documents the retrieval pipeline finds."""

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
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Question: {question}\n\nDocuments:\n\n{documents_context(hits)}",
            config=types.GenerateContentConfig(
                system_instruction=ANSWER_INSTRUCTIONS,
                response_mime_type="application/json",
                response_schema=CitedAnswer,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            ),
        )
        self.usage.add(response)
        parsed = response.parsed
        cited = (
            parsed
            if isinstance(parsed, CitedAnswer)
            else CitedAnswer(answer="", cited_documents=[])
        )

        return SystemAnswer(
            answer=cited.answer.strip() or NOT_FOUND_ANSWER,
            retrieved_doc_ids=list(dict.fromkeys(hit.doc_id for hit in hits)),
            cited_doc_ids=cited_doc_ids(cited.cited_documents, hits),
        )
