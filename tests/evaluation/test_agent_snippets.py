"""Tests for the snippet agent: keyword results as snippets, full reads by id, two budgets."""

from types import SimpleNamespace
from typing import cast

from google import genai
from google.genai import types
from test_agent import FakeModels, final_answer, hit, search_call

from train_rag.evaluation.agent_snippets import READ_LIMIT_REACHED, SnippetAgentSystem
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.retrieval import SearchHit

FULL_TEXT = "Kernel stability\n" + "\n".join(f"line {n}" for n in range(30)) + "\npass rate 0.92"


def read_call(doc_id: str) -> types.GenerateContentResponse:
    """Build a Gemini turn that reads one document.

    Args:
        doc_id: The document to read.

    Returns:
        The response.
    """
    call = types.FunctionCall(name="read_document", args={"doc_id": doc_id})

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(function_call=call)])
            )
        ]
    )


class FakeSnippetBase:
    """Keyword searches find two documents; every known document has FULL_TEXT."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls: list[str] = []

    def retrieve(self, question: str, limit: int) -> list[SearchHit]:
        """Record a full search and return one hit named after it."""
        self.calls.append(f"search: {question}")

        return [hit(question)]

    def retrieve_keywords(self, query: str, limit: int) -> list[SearchHit]:
        """Record a keyword search and its limit; return two hits."""
        self.calls.append(f"keywords: {query} ({limit})")

        return [hit("doc-a"), hit("doc-b")]

    def read_documents(self, doc_ids: list[str]) -> list[SearchHit]:
        """Return FULL_TEXT for every id except 'missing'."""
        self.calls.append(f"read: {','.join(doc_ids)}")

        return [
            hit(doc_id).model_copy(update={"text": FULL_TEXT})
            for doc_id in doc_ids
            if doc_id != "missing"
        ]


def snippet_agent(
    responses: list[types.GenerateContentResponse], max_reads: int = 3
) -> tuple[SnippetAgentSystem, FakeModels, FakeSnippetBase]:
    """Build a snippet agent on a fake Gemini and a fake knowledge base.

    Args:
        responses: Gemini's turns, in order.
        max_reads: Full reads allowed.

    Returns:
        The agent, the fake Gemini and the fake knowledge base.
    """
    models = FakeModels(responses)
    base = FakeSnippetBase()
    client = cast(genai.Client, SimpleNamespace(models=models))
    system = SnippetAgentSystem(base, client, UsageMeter(), max_reads=max_reads)

    return system, models, base


def last_response(models: FakeModels, index: int = 0) -> dict[str, object]:
    """The tool response the model received last, at a position in its turn."""
    parts = models.contents[-1][-1].parts
    assert parts is not None
    response = parts[index].function_response
    assert response is not None and response.response is not None

    return response.response


def test_keyword_search_returns_snippets_of_ten_results() -> None:
    """Keyword results are shown as snippets with ids, not full text, ten at a time."""
    system, models, base = snippet_agent(
        [search_call("pass rate", tool="keyword_search"), final_answer("0.92")]
    )

    result = system.answer("question")

    assert base.calls[1:] == ["keywords: pass rate (10)", "read: doc-a,doc-b"]
    output = str(last_response(models)["results"])
    assert "(id: doc-a," in output and "(id: doc-b," in output
    assert "pass rate 0.92" in output
    assert "line 5" not in output
    assert result.retrieved_doc_ids == ["question", "doc-a", "doc-b"]


def test_read_document_returns_the_full_text_and_counts_as_found() -> None:
    """A read returns the whole document and adds it to the reported documents."""
    system, models, _base = snippet_agent([read_call("doc-z"), final_answer("0.92")])

    result = system.answer("question")

    document = str(last_response(models)["document"])
    assert "line 5" in document and "pass rate 0.92" in document
    assert result.retrieved_doc_ids == ["question", "doc-z"]
    assert result.trace[-2].tool == "read_document"


def test_unknown_id_is_an_error_and_costs_no_read() -> None:
    """Reading an id that doesn't exist says so and leaves the read budget alone."""
    system, models, _base = snippet_agent(
        [read_call("missing"), read_call("doc-z"), final_answer("ok")], max_reads=1
    )

    system.answer("question")

    assert "document" in last_response(models)
    assert models.modes[-1] == types.FunctionCallingConfigMode.AUTO  # searches still left


def test_reads_over_the_limit_are_refused() -> None:
    """Reads past max_reads get an error instead of the document."""
    system, models, _base = snippet_agent(
        [read_call("doc-y"), read_call("doc-z"), final_answer("ok")], max_reads=1
    )

    system.answer("question")

    assert last_response(models) == {"error": READ_LIMIT_REACHED}


def test_tools_switch_off_only_when_searches_and_reads_are_spent() -> None:
    """With four searches and one read used, the next turn can't call tools."""
    system, models, _base = snippet_agent(
        [
            search_call("a", "b", "c", "d", tool="keyword_search"),
            read_call("doc-a"),
            final_answer("done"),
        ],
        max_reads=1,
    )

    system.answer("question")

    assert models.modes == [
        types.FunctionCallingConfigMode.AUTO,
        types.FunctionCallingConfigMode.AUTO,
        types.FunctionCallingConfigMode.NONE,
    ]
