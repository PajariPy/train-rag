"""An agent that leans on cheap keyword searches: few meaning searches, many keyword ones.

Like the snippet agent, keyword results come as matching lines with document ids, and a
tool reads a document in full. The budget is split: MAX_SEMANTIC general searches after the
first, automatic one, and MAX_KEYWORD keyword searches, so the agent can try many wordings.
Its instructions tell it to paraphrase when a keyword search finds nothing relevant. It
caches the conversation's start with Gemini (see agent_cache.py).
"""

from typing import ClassVar

from google import genai

from train_rag.evaluation.agent import Budget
from train_rag.evaluation.agent_snippets import (
    KEYWORD_RESULTS,
    MAX_READS,
    SnippetAgentSystem,
    SnippetKnowledgeBase,
)
from train_rag.evaluation.agent_tools import KEYWORD_TOOL, READ_TOOL
from train_rag.evaluation.system import NOT_FOUND_ANSWER
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.retrieval import SearchHit
from train_rag.prompts import prompt

MAX_SEMANTIC = 2  # general searches after the first, automatic one
MAX_KEYWORD = 10
SEMANTIC_LIMIT = "No general searches left: use keyword_search or read_document, or answer."
KEYWORD_LIMIT = "No keyword searches left: answer from the documents already found."


class KeywordAgentSystem(SnippetAgentSystem):
    """The snippet agent with separate budgets for general and keyword searches."""

    name: ClassVar[str] = "agent-keywords"

    def __init__(
        self,
        knowledge_base: SnippetKnowledgeBase,
        client: genai.Client,
        usage: UsageMeter,
        top_k: int = 5,
        max_semantic: int = MAX_SEMANTIC,
        max_keyword: int = MAX_KEYWORD,
        max_reads: int = MAX_READS,
        keyword_results: int = KEYWORD_RESULTS,
        cache: bool = True,
    ) -> None:
        """Set up the agent.

        Args:
            knowledge_base: The searches and full reads behind the tools.
            client: Gemini client.
            usage: Counts the tokens and searches.
            top_k: Documents the general search returns.
            max_semantic: General searches allowed after the first, automatic one.
            max_keyword: Keyword searches allowed.
            max_reads: Documents the agent may read in full.
            keyword_results: Documents each keyword search shows as snippets.
            cache: Cache the conversation's start with Gemini: the first documents, and
                again after each general search.
        """
        super().__init__(
            knowledge_base,
            client,
            usage,
            top_k,
            1 + max_semantic,
            max_reads,
            keyword_results,
            cache,
        )
        self.max_keyword = max_keyword
        self.instructions = prompt(
            "keyword_agent_instructions",
            max_semantic=max_semantic,
            max_keyword=max_keyword,
            max_reads=max_reads,
            keyword_results=keyword_results,
            not_found=NOT_FOUND_ANSWER,
        )

    def can_call(self, budget: Budget) -> bool:
        """Whether any tool still has budget left.

        Args:
            budget: What the question has used so far.

        Returns:
            True while general searches, keyword searches or reads are left.
        """
        return (
            budget.searches < self.max_searches
            or budget.keyword_searches < self.max_keyword
            or budget.reads < self.max_reads
        )

    def run_tool(
        self, tool: str, arguments: dict[str, object], question: str, budget: Budget
    ) -> tuple[dict[str, str], list[SearchHit]]:
        """Run one tool call within that tool's own budget.

        Args:
            tool: Name of the called tool.
            arguments: Its arguments.
            question: The question, the query when a search call gives none.
            budget: What the question has used so far; updated.

        Returns:
            What the model receives and the documents the call showed it.
        """
        if tool == READ_TOOL:
            return self.read(str(arguments.get("doc_id") or ""), budget)
        if tool == KEYWORD_TOOL:
            if budget.keyword_searches >= self.max_keyword:
                return {"error": KEYWORD_LIMIT}, []
            budget.keyword_searches += 1

            return self.keyword_snippets(str(arguments.get("query") or question))
        if budget.searches >= self.max_searches:
            return {"error": SEMANTIC_LIMIT}, []

        return super().run_tool(tool, arguments, question, budget)
