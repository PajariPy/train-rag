"""Tests for caching the start of the agent's conversation with Gemini."""

from types import SimpleNamespace
from typing import Any, cast

from google import genai
from google.genai import types
from test_agent import final_answer, search_call
from test_agent_snippets import FakeSnippetBase

from train_rag.evaluation.agent_keywords import KeywordAgentSystem
from train_rag.evaluation.usage import UsageMeter


class FakeCaches:
    """Records every cache created and deleted; can be told to refuse creating them."""

    def __init__(self, refuse: bool = False) -> None:
        """Start with no caches.

        Args:
            refuse: Raise on create, like an API error or a too-small cache.
        """
        self.refuse = refuse
        self.created: list[list[types.Content]] = []
        self.deleted: list[str] = []

    def create(self, model: str, config: types.CreateCachedContentConfig) -> Any:  # noqa: ANN401
        """Record a cache of the given turns and name it cache-N."""
        if self.refuse:
            raise RuntimeError("Cached content is too small")
        self.created.append(cast(list[types.Content], config.contents or []))

        return SimpleNamespace(
            name=f"cache-{len(self.created)}",
            usage_metadata=SimpleNamespace(total_token_count=1_000),
        )

    def delete(self, name: str) -> None:
        """Record a deleted cache."""
        self.deleted.append(name)


class FakeCachedModels:
    """Replays prepared responses and records each turn's contents and settings."""

    def __init__(self, responses: list[types.GenerateContentResponse]) -> None:
        """Keep the responses to replay."""
        self.responses = responses
        self.sent: list[list[types.Content]] = []
        self.configs: list[types.GenerateContentConfig] = []

    def generate_content(
        self, model: str, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        """Record the turn and return the next response."""
        self.sent.append(list(contents))
        self.configs.append(config)

        return self.responses.pop(0)


def cached_agent(
    responses: list[types.GenerateContentResponse], refuse: bool = False, max_semantic: int = 2
) -> tuple[KeywordAgentSystem, FakeCachedModels, FakeCaches]:
    """Build the keyword agent (which caches) on a fake Gemini with a cache API."""
    models, caches = FakeCachedModels(responses), FakeCaches(refuse)
    client = cast(genai.Client, SimpleNamespace(models=models, caches=caches))
    system = KeywordAgentSystem(FakeSnippetBase(), client, UsageMeter(), max_semantic=max_semantic)

    return system, models, caches


def text_of(content: types.Content) -> str:
    """The text of a content's parts."""
    return "".join(part.text or "" for part in content.parts or [])


def test_first_documents_are_cached_and_only_the_question_is_sent() -> None:
    """The documents turn goes into a cache; the first call sends just the question."""
    system, models, caches = cached_agent([final_answer("From the first search.")])

    system.answer("How do I set up the VPN?")

    assert len(caches.created) == 1
    assert "text about How do I set up the VPN?" in text_of(caches.created[0][0])
    assert models.configs[0].cached_content == "cache-1"
    assert models.configs[0].system_instruction is None
    assert [text_of(c) for c in models.sent[0]] == ["Question: How do I set up the VPN?"]
    assert caches.deleted == ["cache-1"]


def test_keyword_results_are_sent_in_full_but_not_cached() -> None:
    """After a keyword search the same cache is used; the new turns travel in full."""
    system, models, caches = cached_agent(
        [search_call("pass rate", tool="keyword_search"), final_answer("0.92")]
    )

    system.answer("question")

    assert len(caches.created) == 1
    assert models.configs[1].cached_content == "cache-1"
    assert len(models.sent[1]) == 3  # the question, the model's call, the snippets


def test_a_general_search_is_cached_one_turn_later() -> None:
    """Results of a general search join a new cache once a later turn exists."""
    system, models, caches = cached_agent(
        [
            search_call("vpn client setup"),
            search_call("WG-1042", tool="keyword_search"),
            final_answer("Use WireGuard."),
        ]
    )

    system.answer("question")

    assert len(caches.created) == 2
    assert len(caches.created[1]) == 4  # documents, question, the call, its results
    assert models.configs[1].cached_content == "cache-1"  # the turn right after the search
    assert models.configs[2].cached_content == "cache-2"
    assert len(models.sent[2]) == 2  # only the keyword call and its snippets
    assert caches.deleted == ["cache-1", "cache-2"]


def test_last_turn_without_budget_goes_uncached_with_tools_off() -> None:
    """Once every budget is spent, the answering turn is sent in full with tools off."""
    models, caches = FakeCachedModels([search_call("a"), final_answer("done")]), FakeCaches()
    client = cast(genai.Client, SimpleNamespace(models=models, caches=caches))
    system = KeywordAgentSystem(
        FakeSnippetBase(), client, UsageMeter(), max_semantic=1, max_keyword=0, max_reads=0
    )

    system.answer("question")

    last = models.configs[-1]
    assert last.cached_content is None
    assert last.tool_config is not None
    assert last.tool_config.function_calling_config is not None
    assert last.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.NONE
    assert len(models.sent[-1]) == 4


def test_a_refused_cache_falls_back_to_uncached_calls() -> None:
    """If Gemini won't make the cache, the agent sends the whole conversation itself."""
    system, models, caches = cached_agent([final_answer("ok")], refuse=True)

    result = system.answer("question")

    assert result.answer == "ok"
    assert models.configs[0].cached_content is None
    assert models.configs[0].system_instruction is not None
    assert len(models.sent[0]) == 2
    assert caches.deleted == []


def test_cache_is_deleted_even_when_a_turn_fails() -> None:
    """An error mid-question still deletes the cache."""
    system, _models, caches = cached_agent([])  # no response to give: the first call fails

    try:
        system.answer("question")
    except IndexError:
        pass

    assert caches.deleted == ["cache-1"]


def test_usage_prices_cached_tokens_apart() -> None:
    """Cached tokens are counted apart from input and priced at the cached rate."""
    usage = UsageMeter()
    usage.add(
        types.GenerateContentResponse(
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1_000_000, cached_content_token_count=900_000
            )
        )
    )
    usage.add_cache_write(900_000, 3_600)

    assert usage.input_tokens == 100_000
    assert usage.cached_tokens == 900_000
    # 100k input + 900k written at $0.75, 900k cached at $0.075, 900k stored an hour at $1.
    assert round(usage.cost(), 4) == round(0.75 + 0.0675 + 0.9, 4)
