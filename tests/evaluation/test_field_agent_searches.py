"""Tests for the agent's short ids, field-search filters and the candidate list."""

from field_agent_fakes import FakeRetriever, FakeRewriter, field_index

from train_rag.evaluation.field_agent.searches import AgentSearches
from train_rag.models.field_agent import ToolCall


def searches(mined: list[str]) -> AgentSearches:
    """Build the searches over a small index.

    Args:
        mined: Mined documents, best field-search match first.

    Returns:
        The searches.
    """
    return AgentSearches(FakeRetriever({}, preferred=[]), field_index(mined), FakeRewriter())


def test_ids_resolve_and_unknown_ones_are_dropped() -> None:
    """Short ids map back to dataset ids; unknown ids, brackets and repeats are handled."""
    agent = searches(["m1", "m2"])
    agent.run(ToolCall(name="search_fields", args={"query": "x", "profile": "subject"}))
    assert agent.resolve(["[d2]", "d9", "d2", "d1", 3]) == ["m2", "m1"]


def test_field_search_drops_unknown_filter_values() -> None:
    """An unknown area or channel is ignored rather than filtering everything out."""
    agent = searches(["m1"])
    call = ToolCall(
        name="search_fields",
        args={"query": "x", "profile": "nope", "areas": ["made up"], "channels": ["#nowhere"]},
    )
    output, shown = agent.run(call)
    assert shown == ["m1"]
    assert agent.records[0].profile == "all"
    assert agent.records[0].areas == []
    assert agent.records[0].channels == []
    assert output.startswith("[d1] (chat thread #billing) topic of m1")


def test_candidates_put_picks_first() -> None:
    """The agent's picks lead the candidate list, then every search's results fused."""
    agent = searches(["m1", "m2", "m3"])
    agent.run(ToolCall(name="search_fields", args={"query": "x", "profile": "subject"}))
    candidates = [document.doc_id for document in agent.candidates(["m3"])]
    assert candidates == ["m3", "m1", "m2"]


def test_an_empty_query_is_refused() -> None:
    """A search without a query doesn't run and doesn't use the budget."""
    agent = searches(["m1"])
    output, shown = agent.run(ToolCall(name="search_documents", args={"query": " "}))
    assert (output, shown, agent.records) == ("Bad call: give a query.", [], [])
