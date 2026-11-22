"""Tests for the keyword query: the question with Gemini's rewrites appended."""

from types import SimpleNamespace
from typing import cast

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from train_rag.evaluation.usage import UsageMeter
from train_rag.models.rewrite import Rewrites
from train_rag.query_rewrite import QueryRewriter


class FakeModels:
    """Returns prepared rewrites, or raises an API error."""

    def __init__(self, queries: list[str] | None) -> None:
        """Keep the reply.

        Args:
            queries: Rewrites to return, or None to fail.
        """
        self.queries = queries

    def generate_content(
        self, model: str, contents: str, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        """Return the rewrites as a parsed response, or fail.

        Args:
            model: Ignored.
            contents: Ignored.
            config: Ignored.

        Returns:
            The response.

        Raises:
            APIError: When no rewrites were prepared.
        """
        if self.queries is None:
            raise genai_errors.APIError(503, {"error": {"message": "unavailable"}})
        response = types.GenerateContentResponse()
        response.parsed = Rewrites(queries=self.queries)

        return response


def rewriter(queries: list[str] | None) -> QueryRewriter:
    """Build a rewriter on a fake Gemini.

    Args:
        queries: What Gemini returns, or None to fail.

    Returns:
        The rewriter.
    """
    client = cast(genai.Client, SimpleNamespace(models=FakeModels(queries)))

    return QueryRewriter(client, UsageMeter())


def test_keyword_query_appends_at_most_four_rewrites() -> None:
    """The question comes first, then the first four rewrites."""
    query = rewriter(["a", "b", "c", "d", "e"]).keyword_query("why did it fail")
    assert query == "why did it fail a b c d"


def test_keyword_query_falls_back_to_the_question() -> None:
    """When Gemini fails, BM25 searches the question alone."""
    assert rewriter(None).keyword_query("why did it fail") == "why did it fail"
