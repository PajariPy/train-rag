"""Tests for the keyword agent's separate budgets for general and keyword searches."""

from types import SimpleNamespace
from typing import cast

from google import genai
from google.genai import types
from test_agent import FakeModels, final_answer, search_call
from test_agent_snippets import FakeSnippetBase, last_response, read_call

from train_rag.evaluation.agent_keywords import (
    KEYWORD_LIMIT,
    SEMANTIC_LIMIT,
    KeywordAgentSystem,
)
from train_rag.evaluation.usage import UsageMeter

AUTO = types.FunctionCallingConfigMode.AUTO
NONE = types.FunctionCallingConfigMode.NONE


def keyword_agent(
    responses: list[types.GenerateContentResponse],
    max_semantic: int = 2,
    max_keyword: int = 10,
    max_reads: int = 3,
) -> tuple[KeywordAgentSystem, FakeModels, FakeSnippetBase]:
    """Build a keyword agent on a fake Gemini and a fake knowledge base.

    Args:
        responses: Gemini's turns, in order.
        max_semantic: General searches allowed after the first.
        max_keyword: Keyword searches allowed.
        max_reads: Full reads allowed.

    Returns:
        The agent, the fake Gemini and the fake knowledge base.
    """
    models = FakeModels(responses)
    base = FakeSnippetBase()
    client = cast(genai.Client, SimpleNamespace(models=models))
    system = KeywordAgentSystem(
        base, client, UsageMeter(), max_semantic=max_semantic, max_keyword=max_keyword,
        max_reads=max_reads,
    )

    return system, models, base


def test_general_searches_run_out_while_keyword_searches_go_on() -> None:
    """After 2 general searches a third is refused, but keyword searches still run."""
    system, models, base = keyword_agent(
        [
            search_call("a", "b", "c"),
            search_call("pass rate", tool="keyword_search"),
            final_answer("0.92"),
        ]
    )

    system.answer("question")

    assert base.calls[1:3] == ["search: a", "search: b"]
    assert "keywords: pass rate (10)" in base.calls
    assert models.modes == [AUTO, AUTO, AUTO]


def test_refused_general_search_points_to_the_keyword_search() -> None:
    """A general search over the budget gets an error naming the tools still left."""
    system, models, _base = keyword_agent(
        [search_call("a", "b"), final_answer("ok")], max_semantic=1
    )

    system.answer("question")

    assert last_response(models, 1) == {"error": SEMANTIC_LIMIT}


def test_keyword_searches_are_capped_at_their_own_budget() -> None:
    """Keyword searches past max_keyword are refused; the tools then switch off."""
    system, models, base = keyword_agent(
        [search_call("w1", "w2", "w3", tool="keyword_search"), final_answer("done")],
        max_semantic=0,
        max_keyword=2,
        max_reads=0,
    )

    system.answer("question")

    assert [call for call in base.calls if call.startswith("keywords")] == [
        "keywords: w1 (10)",
        "keywords: w2 (10)",
    ]
    assert last_response(models, 2) == {"error": KEYWORD_LIMIT}
    assert models.modes == [AUTO, NONE]


def test_reads_keep_their_own_budget() -> None:
    """A read works with keyword budget left and counts against the read budget only."""
    system, models, _base = keyword_agent([read_call("doc-z"), final_answer("0.92")])

    result = system.answer("question")

    assert "document" in last_response(models)
    assert result.retrieved_doc_ids == ["question", "doc-z"]


def test_instructions_give_both_budgets_and_ask_to_paraphrase() -> None:
    """The instructions state each budget and tell the agent to reword unmatched searches."""
    system, _models, _base = keyword_agent([final_answer("x")])

    assert "up to 10 times" in system.instructions
    assert "up to 2 times" in system.instructions
    assert "paraphrase" in system.instructions
