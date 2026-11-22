"""Tests for the field-search agent: both search tools, the budget, the check and asking."""

from field_agent_fakes import FakeGemini, FakeRetriever, FakeRewriter, field_index, turn, usage

from train_rag.evaluation.field_agent.system import ClarifyingFieldAgent, FieldAgentSystem
from train_rag.evaluation.registry import CLARIFYING_SYSTEMS, SYSTEMS
from train_rag.prompts import prompt

QUESTION = "How did support resolve the duplicate charges?"
FIELDS = ("search_fields", {"query": "duplicate charges", "profile": "subject"})


class FakePerson:
    """Replies the same thing to every question and records the questions."""

    def __init__(self) -> None:
        """Start with no questions."""
        self.asks: list[str] = []

    def reply(self, ask: str) -> str:
        """Record the question and reply.

        Args:
            ask: The agent's question.

        Returns:
            The reply.
        """
        self.asks.append(ask)

        return "the billing team, last spring"


def system(
    gemini: FakeGemini,
    retriever: FakeRetriever,
    mined: list[str],
    clarifying: bool = False,
) -> tuple[FieldAgentSystem, FakeRewriter]:
    """Build the agent on fakes.

    Args:
        gemini: The fake Gemini.
        retriever: The fake pipeline.
        mined: The mined documents, best field-search match first.
        clarifying: Build the asking agent.

    Returns:
        The system and its rewriter.
    """
    kind = ClarifyingFieldAgent if clarifying else FieldAgentSystem
    agent = kind(retriever, field_index(mined), gemini.client(), usage())
    rewriter = FakeRewriter()
    agent.rewriter = rewriter

    return agent, rewriter


def test_both_tools_feed_the_final_rerank_and_the_cited_answer() -> None:
    """Picks come first among the candidates; fitting documents lead the answer."""
    gemini = FakeGemini(
        [
            turn(FIELDS),
            turn(("search_documents", {"query": "refund email"})),
            turn(("finish", {"ids": ["d1", "d4"]})),
            turn(("judge_cases", {"reasoning": "one", "verdict": "same_answer", "fits": ["d4"]})),
        ],
        answer="Refunded [1].",
    )
    retriever = FakeRetriever({"refund email": ["g10", "m2"]}, preferred=["m1", "g10"])
    agent, rewriter = system(gemini, retriever, ["m1", "m2", "m3"])

    result = agent.answer(QUESTION)

    assert rewriter.queries == ["refund email"]
    assert retriever.searches == [("refund email", "refund email rewritten")]
    query, candidates = retriever.reranked[-1]
    assert query == QUESTION
    assert candidates[:2] == ["m1", "g10"]
    assert set(candidates) == {"m1", "m2", "m3", "g10"}
    assert result.retrieved_doc_ids == ["g10", "m1", "m2", "m3"]
    assert result.answer == "Refunded [g10]."
    assert "ask_user" not in gemini.allowed[0]
    assert gemini.allowed[3] == ["judge_cases"]
    assert "judge_cases" in str(result.trace[-1].tool_calls)


def test_search_budget_is_shared_with_at_most_two_general_searches() -> None:
    """A third general search and a fifth search are refused; then only finish is offered."""
    general = ("search_documents", {"query": "q"})
    gemini = FakeGemini(
        [
            turn(general),
            turn(general),
            turn(general),
            turn(FIELDS),
            turn(FIELDS),
            turn(FIELDS),
            turn(("finish", {"ids": []})),
            turn(("judge_cases", {"reasoning": "", "verdict": "same_answer", "fits": []})),
        ],
        answer="Answer [1].",
    )
    retriever = FakeRetriever({"q": ["g1"]}, preferred=[])
    agent, _rewriter = system(gemini, retriever, ["m1"])

    result = agent.answer(QUESTION)

    assert len(retriever.searches) == 2
    assert "search_documents" not in gemini.allowed[2]
    assert gemini.allowed[6] == ["finish"]
    refused = [step.output for step in result.trace if step.kind == "tool"]
    assert refused[2].startswith("No search_documents left")
    assert refused[5].startswith("No search_fields left")


def test_general_search_reaches_unmined_documents() -> None:
    """A document without mined fields reaches the answer through search_documents."""
    gemini = FakeGemini(
        [
            turn(("search_documents", {"query": "contract email"})),
            turn(("finish", {"ids": ["d1"]})),
            turn(("judge_cases", {"reasoning": "", "verdict": "same_answer", "fits": ["d1"]})),
        ],
        answer="Signed [1].",
    )
    retriever = FakeRetriever({"contract email": ["g20"]}, preferred=[])
    agent, _rewriter = system(gemini, retriever, ["m1"])

    result = agent.answer(QUESTION)

    assert result.retrieved_doc_ids == ["g20"]
    shown = next(step.output for step in result.trace if step.tool == "search_documents")
    assert "(no mined fields)" in shown


def test_several_cases_without_a_person_are_shown_side_by_side() -> None:
    """With no one to ask, the answer is told to present the cases side by side."""
    gemini = FakeGemini(
        [
            turn(FIELDS),
            turn(("finish", {"ids": ["d1", "d2"]})),
            turn(("judge_cases", {"reasoning": "two", "verdict": "several_cases_fit"})),
        ],
        answer="Two cases [1].",
    )
    agent, _rewriter = system(gemini, FakeRetriever({}, preferred=[]), ["m1", "m2"])

    agent.answer(QUESTION)

    assert prompt("side_by_side_note") in gemini.answer_prompts[0]


def test_several_cases_make_the_agent_ask_then_check_again() -> None:
    """The first check asks the person; the second, still several, ends side by side."""
    several = ("judge_cases", {"reasoning": "two teams", "verdict": "several_cases_fit"})
    gemini = FakeGemini(
        [
            turn(FIELDS),
            turn(("finish", {"ids": ["d1"]})),
            turn(several),
            turn(("ask_user", {"question": "Which team was it?"})),
            turn(FIELDS),
            turn(("finish", {"ids": ["d2"]})),
            turn(several),
        ],
        answer="Answer [1].",
    )
    agent, _rewriter = system(gemini, FakeRetriever({}, preferred=[]), ["m1", "m2"], True)
    person = FakePerson()

    result = agent.answer_with(QUESTION, person)

    assert person.asks == ["Which team was it?"]
    assert "ask_user" in gemini.allowed[3]
    assert sum("judge_cases" in str(step.tool_calls) for step in result.trace) == 2
    assert prompt("side_by_side_note") in gemini.answer_prompts[0]


def test_field_agents_are_registered_and_only_the_asking_one_needs_notes() -> None:
    """Both systems can be run by name; memory notes are checked for the asking one."""
    assert {FieldAgentSystem.name, ClarifyingFieldAgent.name} <= set(SYSTEMS)
    assert ClarifyingFieldAgent.name in CLARIFYING_SYSTEMS
    assert FieldAgentSystem.name not in CLARIFYING_SYSTEMS
