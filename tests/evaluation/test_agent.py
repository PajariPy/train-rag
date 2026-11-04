"""Tests for the ReAct agent's search loop, with a fake Gemini and a fake knowledge base."""

from types import SimpleNamespace
from typing import Any, cast

from google import genai
from google.genai import types

from train_rag.evaluation.agent import LIMIT_REACHED, AgentSystem
from train_rag.evaluation.usage import UsageMeter
from train_rag.models import SourceType
from train_rag.models.retrieval import SearchHit


def search_call(
    *queries: str, tool: str = "search_knowledge_base"
) -> types.GenerateContentResponse:
    """Build a Gemini turn that calls a search tool once per query.

    Args:
        queries: Search queries.
        tool: The called tool.

    Returns:
        The response.
    """
    parts = [
        types.Part(function_call=types.FunctionCall(name=tool, args={"query": query}))
        for query in queries
    ]

    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))]
    )


def final_answer(text: str) -> types.GenerateContentResponse:
    """Build a Gemini turn that answers.

    Args:
        text: The answer.

    Returns:
        The response.
    """
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))
        ]
    )


def hit(doc_id: str) -> SearchHit:
    """Build one search hit.

    Args:
        doc_id: Its dataset id, also used in its text.

    Returns:
        The hit.
    """
    return SearchHit(
        doc_id=doc_id,
        document_id=1,
        source_type=SourceType.SLACK,
        field="fused",
        text=f"text about {doc_id}",
        distance=0.0,
    )


class FakeModels:
    """Replays prepared responses and records each turn's tool mode and contents."""

    def __init__(self, responses: list[types.GenerateContentResponse]) -> None:
        """Keep the responses to replay.

        Args:
            responses: One response per turn, in order.
        """
        self.responses = responses
        self.modes: list[Any] = []
        self.contents: list[list[types.Content]] = []

    def generate_content(
        self, model: str, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        """Return the next response.

        Args:
            model: Ignored.
            contents: The conversation so far.
            config: The turn's settings.

        Returns:
            The next prepared response.
        """
        assert config.tool_config is not None
        assert config.tool_config.function_calling_config is not None
        self.modes.append(config.tool_config.function_calling_config.mode)
        self.contents.append(list(contents))

        return self.responses.pop(0)


class FakeKnowledgeBase:
    """Returns one document per search, named after the query, and records every search."""

    def __init__(self) -> None:
        """Start with no searches."""
        self.queries: list[str] = []

    def retrieve(self, question: str, limit: int) -> list[SearchHit]:
        """Record a full search and return one hit named after it.

        Args:
            question: The query.
            limit: Ignored.

        Returns:
            One hit whose doc id is the query.
        """
        self.queries.append(question)

        return [hit(question)]

    def retrieve_keywords(self, query: str, limit: int) -> list[SearchHit]:
        """Record a keyword search and return one hit named after it.

        Args:
            query: The keywords.
            limit: Ignored.

        Returns:
            One hit whose doc id marks it as a keyword search.
        """
        self.queries.append(f"keywords: {query}")

        return [hit(f"keywords: {query}")]


def agent(
    responses: list[types.GenerateContentResponse],
) -> tuple[AgentSystem, FakeModels, FakeKnowledgeBase]:
    """Build an agent on a fake Gemini and a fake knowledge base.

    Args:
        responses: Gemini's turns, in order.

    Returns:
        The agent, the fake Gemini and the fake knowledge base.
    """
    models = FakeModels(responses)
    knowledge_base = FakeKnowledgeBase()
    client = cast(genai.Client, SimpleNamespace(models=models))

    return AgentSystem(knowledge_base, client, UsageMeter()), models, knowledge_base


def test_agent_answers_from_the_first_search_without_calling_tools() -> None:
    """The question is searched before Gemini's first turn; Gemini can answer at once."""
    system, models, knowledge_base = agent([final_answer("From the first search.")])

    result = system.answer("How do I set up the VPN?")

    assert knowledge_base.queries == ["How do I set up the VPN?"]
    assert result.retrieved_doc_ids == ["How do I set up the VPN?"]
    assert result.answer == "From the first search."
    first_turn = models.contents[0][0].parts
    assert first_turn is not None
    assert "text about How do I set up the VPN?" in str(first_turn[0].text)


def test_agent_uses_both_tools_and_reports_every_document_in_order() -> None:
    """A full search and a keyword search follow the first; all documents are reported."""
    system, models, knowledge_base = agent(
        [
            search_call("vpn client setup"),
            search_call("WG-1042", tool="keyword_search"),
            final_answer("Use WireGuard."),
        ]
    )

    result = system.answer("question")

    assert knowledge_base.queries == ["question", "vpn client setup", "keywords: WG-1042"]
    assert result.retrieved_doc_ids == ["question", "vpn client setup", "keywords: WG-1042"]
    assert models.modes == [
        types.FunctionCallingConfigMode.AUTO,
        types.FunctionCallingConfigMode.AUTO,
        types.FunctionCallingConfigMode.NONE,
    ]


def test_agent_refuses_searches_over_the_limit_in_one_turn() -> None:
    """With the first search counted, only two more run; the rest get an error."""
    system, models, knowledge_base = agent([search_call("a", "b", "c"), final_answer("done")])

    system.answer("question")

    assert knowledge_base.queries == ["question", "a", "b"]
    assert models.modes[-1] == types.FunctionCallingConfigMode.NONE
    last_turn = models.contents[-1][-1]
    assert last_turn.parts is not None
    refused = last_turn.parts[2].function_response
    assert refused is not None
    assert refused.response == {"error": LIMIT_REACHED}


def test_agent_traces_every_step() -> None:
    """The trace holds the instructions, the input, each model turn and each search."""
    system, _models, _knowledge_base = agent(
        [search_call("WG-1042", tool="keyword_search"), final_answer("Use WireGuard.")]
    )

    result = system.answer("question")

    assert [step.kind for step in result.trace] == [
        "instructions",
        "tool",
        "input",
        "model",
        "tool",
        "model",
    ]
    _, automatic, first_turn, call_turn, keyword, final = result.trace
    assert automatic.arguments == {"query": "question", "automatic": True}
    assert automatic.doc_ids == ["question"]
    assert "text about question" in first_turn.text
    assert call_turn.tool_calls == [{"tool": "keyword_search", "arguments": {"query": "WG-1042"}}]
    assert keyword.tool == "keyword_search"
    assert keyword.doc_ids == ["keywords: WG-1042"]
    assert "text about keywords: WG-1042" in keyword.output
    assert final.text == "Use WireGuard."
