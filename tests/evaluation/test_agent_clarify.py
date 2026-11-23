"""Tests for the snippet agent that can ask the person behind the question."""

from types import SimpleNamespace
from typing import cast

from google import genai
from google.genai import types
from test_agent import FakeModels, final_answer, search_call
from test_agent_snippets import FakeSnippetBase

from train_rag.evaluation.agent_clarify import ASK_LIMIT_REACHED, ClarifyingSnippetAgent
from train_rag.evaluation.agent_snippets import SnippetAgentSystem
from train_rag.evaluation.usage import UsageMeter


class FakePerson:
    """Replies the same thing to every question and records the questions."""

    def __init__(self, reply: str = "the inference team") -> None:
        """Start with no questions.

        Args:
            reply: What the person always replies.
        """
        self.asks: list[str] = []
        self.answer = reply

    def reply(self, ask: str) -> str:
        """Record the question and reply."""
        self.asks.append(ask)

        return self.answer


def ask_call(question: str) -> types.GenerateContentResponse:
    """Build a Gemini turn that asks the person one question.

    Args:
        question: What the agent asks.

    Returns:
        The response.
    """
    call = types.FunctionCall(name="ask_user", args={"question": question})

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(function_call=call)])
            )
        ]
    )


def clarifying_agent(
    responses: list[types.GenerateContentResponse],
) -> tuple[ClarifyingSnippetAgent, FakeSnippetBase]:
    """Build the clarifying agent on a fake Gemini and a fake knowledge base.

    Args:
        responses: Gemini's turns, in order.

    Returns:
        The agent and the fake knowledge base.
    """
    base = FakeSnippetBase()
    client = cast(genai.Client, SimpleNamespace(models=FakeModels(responses)))

    return ClarifyingSnippetAgent(base, client, UsageMeter()), base


def test_ask_reaches_person_and_reply_goes_back() -> None:
    """The agent's question goes to the person, and the reply is recorded in the trace."""
    agent, _ = clarifying_agent([ask_call("Which team?"), final_answer("A")])
    person = FakePerson()

    result = agent.answer_with("q?", person)

    assert person.asks == ["Which team?"]
    assert any(s.tool == "ask_user" and s.output == "the inference team" for s in result.trace)
    assert result.answer == "A"


def test_fourth_ask_is_refused() -> None:
    """After three questions the person isn't asked again; the agent gets the limit message."""
    agent, _ = clarifying_agent([*(ask_call(f"q{n}") for n in range(4)), final_answer("A")])
    person = FakePerson()

    result = agent.answer_with("q?", person)

    assert person.asks == ["q0", "q1", "q2"]
    asks = [step for step in result.trace if step.tool == "ask_user"]
    assert asks[-1].output == ASK_LIMIT_REACHED


def test_asks_do_not_use_search_budget() -> None:
    """Three asks still leave all four extra searches."""
    keyword_turns = [search_call(f"k{n}", tool="keyword_search") for n in range(4)]
    agent, base = clarifying_agent(
        [*(ask_call(f"q{n}") for n in range(3)), *keyword_turns, final_answer("A")]
    )

    agent.answer_with("q?", FakePerson())

    assert [call for call in base.calls if call.startswith("keywords")] == [
        f"keywords: k{n} (10)" for n in range(4)
    ]


def test_without_person_behaves_like_snippet_agent() -> None:
    """Without a person, the agent answers as the snippet agent does."""
    turns = [search_call("x", tool="keyword_search"), final_answer("A")]
    plain = SnippetAgentSystem(
        FakeSnippetBase(),
        cast(genai.Client, SimpleNamespace(models=FakeModels(list(turns)))),
        UsageMeter(),
    )
    agent, _ = clarifying_agent(list(turns))

    assert agent.answer("q?").retrieved_doc_ids == plain.answer("q?").retrieved_doc_ids


def test_ends_after_refused_calls_even_with_asks_left() -> None:
    """With searches used up and asks left, repeated refused searches still force an answer."""
    searches = [search_call(f"k{n}", tool="keyword_search") for n in range(4)]
    refused = [search_call(f"r{n}", tool="keyword_search") for n in range(2)]
    models = FakeModels([*searches, *refused, final_answer("A")])
    agent = ClarifyingSnippetAgent(
        FakeSnippetBase(),
        cast(genai.Client, SimpleNamespace(models=models)),
        UsageMeter(),
        max_reads=0,
    )

    result = agent.answer_with("q?", FakePerson())

    assert result.answer == "A"
    assert models.modes[-1] == types.FunctionCallingConfigMode.NONE


def test_prompt_suggests_asking_before_not_found() -> None:
    """The instructions tell the agent to consider asking before giving up."""
    agent, _ = clarifying_agent([])

    assert "before replying" in agent.instructions.lower()


def test_answered_ask_gives_two_more_searches() -> None:
    """After the searches are used up, an answered question lets the agent search twice more."""
    used_up = [search_call(f"k{n}", tool="keyword_search") for n in range(4)]
    after = [search_call(f"a{n}", tool="keyword_search") for n in range(2)]
    agent, base = clarifying_agent([*used_up, ask_call("Which team?"), *after, final_answer("A")])

    agent.answer_with("q?", FakePerson())

    assert [c for c in base.calls if c.startswith("keywords: a")] == [
        "keywords: a0 (10)",
        "keywords: a1 (10)",
    ]


def test_refused_ask_gives_no_searches() -> None:
    """A question over the ask limit earns no extra searches."""
    agent, base = clarifying_agent(
        [
            *(ask_call(f"q{n}") for n in range(4)),
            *(search_call(f"k{n}", tool="keyword_search") for n in range(11)),
            final_answer("A"),
        ]
    )

    agent.answer_with("q?", FakePerson())

    assert len([c for c in base.calls if c.startswith("keywords")]) == 4 + 2 * 3


def test_prompt_says_ask_early() -> None:
    """The agent is told to ask before spending its searches, and not for names or ids."""
    agent, _ = clarifying_agent([])
    text = agent.instructions.lower()

    assert "before searching further" in text
    assert "kind of document" in text
