"""Rewrite a question into search queries for BM25, the keyword half of the first search.

On the benchmark's 125 semantic questions, BM25 on the question with Gemini's 4 rewrites
appended lifted the reranked top 5 from 64.0% to 67.2%; gte stays on the plain question,
since rewritten vectors hurt (decision 16 Nov, article 006).
"""

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from train_rag.evaluation.pipeline_answer import GEMINI_MODEL
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.rewrite import Rewrites
from train_rag.prompts import prompt

REWRITES = 4


class QueryRewriter:
    """Gemini Flash writing the keyword query of a question."""

    def __init__(self, client: genai.Client, usage: UsageMeter, count: int = REWRITES) -> None:
        """Set up the rewriter.

        Args:
            client: Gemini client.
            usage: Counts the tokens of every rewrite.
            count: Search queries asked for.
        """
        self.client = client
        self.usage = usage
        self.count = count
        self.instructions = prompt("rewrite_instructions", count=count)

    def rewrites(self, question: str) -> list[str]:
        """Ask Gemini for the search queries of one question.

        Args:
            question: The question or the agent's query.

        Returns:
            Up to `count` queries, keyword query first; none when Gemini fails.
        """
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"Question: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=self.instructions,
                    response_mime_type="application/json",
                    response_schema=Rewrites,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    ),
                ),
            )
        except genai_errors.APIError:
            return []

        self.usage.add(response)
        parsed = response.parsed

        return parsed.queries[: self.count] if isinstance(parsed, Rewrites) else []

    def keyword_query(self, question: str) -> str:
        """The text BM25 searches: the question with its rewrites appended.

        Args:
            question: The question or the agent's query.

        Returns:
            The question and the rewrites, joined by spaces; the question alone when
            there are no rewrites.
        """
        return " ".join([question, *self.rewrites(question)])
