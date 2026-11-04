"""A RAG system that answers from the documents' own summaries, found by vector search."""

from typing import ClassVar

from google import genai
from google.genai import types

from train_rag.evaluation.system import NOT_FOUND_ANSWER, RagSystem
from train_rag.models.evaluation import SystemAnswer
from train_rag.models.retrieval import SearchHit
from train_rag.prompts import prompt
from train_rag.retrieval import Retriever

GEMINI_MODEL = "gemini-3.8-flash"
MAX_CONTEXT_CHARS = 1_500
ANSWER_INSTRUCTIONS = prompt("summary_answer_instructions", not_found=NOT_FOUND_ANSWER)


def context(hits: list[SearchHit]) -> str:
    """Lay out the retrieved texts for the prompt.

    Args:
        hits: Retrieved texts, best first.

    Returns:
        The numbered excerpts, each cut to `MAX_CONTEXT_CHARS`.
    """
    return "\n\n".join(
        f"[{index}] {hit.source_type} {hit.field} ({hit.doc_id}):\n{hit.text[:MAX_CONTEXT_CHARS]}"
        for index, hit in enumerate(hits, 1)
    )


class SummarySearchSystem(RagSystem):
    """Retrieve the closest summaries, then let Gemini answer from them.

    Only documents with a summary are searchable, so the system can only find what those
    summaries say.
    """

    name: ClassVar[str] = "summary-search"

    def __init__(self, retriever: Retriever, client: genai.Client, top_k: int = 10) -> None:
        """Set up the system.

        Args:
            retriever: Finds the summaries for a question.
            client: Gemini client for the answer.
            top_k: Summaries retrieved and shown to the model.
        """
        self.retriever = retriever
        self.client = client
        self.top_k = top_k

    def answer(self, question: str) -> SystemAnswer:
        """Retrieve summaries and answer from them.

        Args:
            question: The question text.

        Returns:
            The answer and the retrieved documents, best first.
        """
        hits = self.retriever.retrieve(question, self.top_k)
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Question: {question}\n\nExcerpts:\n\n{context(hits)}",
            config=types.GenerateContentConfig(
                system_instruction=ANSWER_INSTRUCTIONS,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            ),
        )

        return SystemAnswer(
            answer=(response.text or NOT_FOUND_ANSWER).strip(),
            retrieved_doc_ids=list(dict.fromkeys(hit.doc_id for hit in hits)),
        )
