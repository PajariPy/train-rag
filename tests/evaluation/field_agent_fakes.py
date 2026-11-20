"""Fakes for the field-agent tests: Gemini, the retrieval pipeline, the rewriter, an index."""

from collections.abc import Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from google import genai
from google.genai import types

from train_rag.embedder import Embedder
from train_rag.evaluation.pipeline_answer import CitedAnswer
from train_rag.evaluation.usage import UsageMeter
from train_rag.mined_fields.index import FieldIndex
from train_rag.models import SourceType
from train_rag.models.retrieval import RetrievedDocument, SearchHit
from train_rag.query_rewrite import QueryRewriter
from train_rag.tables import DocumentFieldsRecord, FieldVectorRecord

DIMENSIONS = 768


def turn(*calls: tuple[str, dict[str, Any]]) -> types.GenerateContentResponse:
    """Build a Gemini turn that makes the given tool calls.

    Args:
        calls: (tool name, arguments) of each call.

    Returns:
        The response.
    """
    parts = [
        types.Part(function_call=types.FunctionCall(name=name, args=args)) for name, args in calls
    ]

    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))]
    )


class FakeGemini:
    """Replays the agent's turns and answers the answer call with a fixed cited answer."""

    def __init__(self, turns: list[types.GenerateContentResponse], answer: str) -> None:
        """Keep the turns and the answer.

        Args:
            turns: The agent's turns, in order.
            answer: The answer text, with inline [n] citations.
        """
        self.turns = turns
        self.answer = answer
        self.allowed: list[list[str]] = []
        self.answer_prompts: list[str] = []
        self.models = self

    def client(self) -> genai.Client:
        """This fake as a Gemini client.

        Returns:
            The client.
        """
        return cast(genai.Client, SimpleNamespace(models=self))

    def generate_content(
        self, model: str, contents: object, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        """Return the next agent turn, or the answer for the answer call.

        Args:
            model: Ignored.
            contents: The conversation, or the answer prompt.
            config: The call's settings.

        Returns:
            The response.
        """
        if config.response_schema is CitedAnswer:
            self.answer_prompts.append(str(contents))
            response = types.GenerateContentResponse()
            response.parsed = CitedAnswer(answer=self.answer, cited_documents=[1])
            return response

        assert config.tool_config is not None
        calling = config.tool_config.function_calling_config
        assert calling is not None
        self.allowed.append(list(calling.allowed_function_names or []))

        return self.turns.pop(0)


class FakeRewriter(QueryRewriter):
    """Appends one fixed word and records the queries it rewrote."""

    def __init__(self) -> None:
        """Start with no queries."""
        self.queries: list[str] = []

    def keyword_query(self, question: str) -> str:
        """Record the query and append a word.

        Args:
            question: The query.

        Returns:
            The query with "rewritten" appended.
        """
        self.queries.append(question)

        return f"{question} rewritten"


class FakeRetriever:
    """General search returning fixed documents; reranks by a fixed preference order."""

    def __init__(self, results: dict[str, list[str]], preferred: list[str]) -> None:
        """Keep the results and the reranker's order.

        Args:
            results: Doc ids each general-search query returns, best first.
            preferred: Doc ids the reranker puts first, in order; others follow.
        """
        self.results = results
        self.preferred = preferred
        self.searches: list[tuple[str, str | None]] = []
        self.reranked: list[tuple[str, list[str]]] = []

    def search(
        self, query: str, limit: int, keyword_query: str | None = None
    ) -> list[RetrievedDocument]:
        """Return the query's fixed documents.

        Args:
            query: The agent's query.
            limit: Ignored.
            keyword_query: Recorded.

        Returns:
            The documents, with title and text.
        """
        self.searches.append((query, keyword_query))

        return [
            RetrievedDocument(
                doc_id=doc_id,
                document_id=number(doc_id),
                source_type=SourceType.GMAIL,
                score=0.0,
                ranks={},
                title=f"title of {doc_id}",
                text=f"text of {doc_id}",
            )
            for doc_id in self.results.get(query, [])
        ]

    def rerank(
        self, query: str, documents: list[RetrievedDocument], scores: dict[str, float]
    ) -> list[RetrievedDocument]:
        """Order documents: preferred ones first, then as given.

        Args:
            query: Recorded.
            documents: The candidates.
            scores: Filled with each document's score.

        Returns:
            The documents, most relevant first.
        """
        self.reranked.append((query, [document.doc_id for document in documents]))
        for place, document in enumerate(documents):
            if document.doc_id in self.preferred:
                scores[document.doc_id] = 100.0 - self.preferred.index(document.doc_id)
            else:
                scores.setdefault(document.doc_id, -float(place))

        return sorted(documents, key=lambda document: scores[document.doc_id], reverse=True)

    def read_documents(self, doc_ids: list[str]) -> list[SearchHit]:
        """Return each document's full text.

        Args:
            doc_ids: Dataset ids.

        Returns:
            One hit per id, in order.
        """
        return [
            SearchHit(
                doc_id=doc_id,
                document_id=number(doc_id),
                source_type=SourceType.SLACK,
                field="document",
                text=f"full text of {doc_id}",
                distance=0.0,
            )
            for doc_id in doc_ids
        ]


def number(doc_id: str) -> int:
    """The row id of a test document named like "m3" or "g12".

    Args:
        doc_id: The dataset id.

    Returns:
        Its digits as a number.
    """
    return int("".join(character for character in doc_id if character.isdigit()) or 0)


class AxisEmbedder(Embedder):
    """Embeds every text as the unit vector along axis 0."""

    name = "axis"
    dimensions = DIMENSIONS
    max_tokens = 10

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed each text.

        Args:
            texts: Any texts.

        Returns:
            One vector per text.
        """
        vector = np.zeros(DIMENSIONS)
        vector[0] = 1.0

        return [vector.tolist() for _ in texts]


def field_index(doc_ids: list[str]) -> FieldIndex:
    """Build an index of mined Slack threads, best match first.

    Args:
        doc_ids: The mined documents, in the order a search ranks them.

    Returns:
        The index.
    """
    rows = []
    for place, doc_id in enumerate(doc_ids):
        vector = np.zeros(DIMENSIONS)
        vector[0] = 1.0 - place / 100
        vector[1] = np.sqrt(1 - vector[0] ** 2)
        fields = DocumentFieldsRecord(
            id=number(doc_id),
            document_id=number(doc_id),
            doc_id=doc_id,
            source_type="slack",
            extractor="deepseek-test",
            raw={},
            fields={"topic": [f"topic of {doc_id}"]},
            areas=["billing and pricing"],
            channel="billing",
            created_at=datetime(2026, 11, 1, tzinfo=UTC),
        )
        vectors = FieldVectorRecord(
            document_fields_id=number(doc_id), model="axis", subject=vector.tolist()
        )
        rows.append((fields, vectors))

    return FieldIndex(rows, AxisEmbedder())


def usage() -> UsageMeter:
    """A fresh usage meter.

    Returns:
        The meter.
    """
    return UsageMeter()
