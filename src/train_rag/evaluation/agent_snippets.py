"""An agent whose keyword search returns snippets, with a tool to read a document in full.

Full documents are expensive in tokens. Here the keyword search shows each of its results as
the lines that match the query, with a little context and the document's id, so the agent
can scan twice as many results for fewer tokens and read in full, by id, only the ones that
look right. The general search and the first automatic search still return full text.
"""

from typing import ClassVar, Protocol

from google import genai
from google.genai import types

from train_rag.evaluation.agent import AgentSystem, Budget, KnowledgeBase
from train_rag.evaluation.agent_tools import KEYWORD_TOOL, READ_TOOL, snippet_tools
from train_rag.evaluation.snippets import snippet_context
from train_rag.evaluation.system import NOT_FOUND_ANSWER
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.retrieval import SearchHit
from train_rag.prompts import prompt

MAX_SEARCHES = 5  # counting the first, automatic one
MAX_READS = 3
KEYWORD_RESULTS = 10
READ_CHARS = 30_000  # about one gte chunk; longer documents are cut
READ_LIMIT_REACHED = "Read limit reached: answer from the documents already read."


class SnippetKnowledgeBase(KnowledgeBase, Protocol):
    """The searches, plus reading documents in full by id."""

    def read_documents(self, doc_ids: list[str]) -> list[SearchHit]:
        """Documents' full text (title on top), in the order asked; unknown ids are left out."""
        ...


class SnippetAgentSystem(AgentSystem):
    """The agent, with keyword results as snippets and a separate budget of full reads."""

    name: ClassVar[str] = "agent-snippets"

    def __init__(
        self,
        knowledge_base: SnippetKnowledgeBase,
        client: genai.Client,
        usage: UsageMeter,
        top_k: int = 5,
        max_searches: int = MAX_SEARCHES,
        max_reads: int = MAX_READS,
        keyword_results: int = KEYWORD_RESULTS,
        cache: bool = False,
    ) -> None:
        """Set up the agent.

        Args:
            knowledge_base: The searches and full reads behind the tools.
            client: Gemini client.
            usage: Counts the tokens and searches.
            top_k: Documents the general search returns.
            max_searches: Searches allowed per question, counting the first one.
            max_reads: Documents the agent may read in full per question.
            keyword_results: Documents the keyword search shows as snippets.
            cache: Cache the conversation's start with Gemini.
        """
        super().__init__(knowledge_base, client, usage, top_k, max_searches, cache)
        self.reader = knowledge_base
        self.max_reads = max_reads
        self.keyword_results = keyword_results
        self.instructions = prompt(
            "snippet_agent_instructions",
            max_searches=max_searches - 1,
            max_reads=max_reads,
            keyword_results=keyword_results,
            not_found=NOT_FOUND_ANSWER,
        )

    def tools(self) -> types.Tool:
        """The general search, the keyword snippets and the full read.

        Returns:
            The tool declarations.
        """
        return snippet_tools()

    def can_call(self, budget: Budget) -> bool:
        """Whether the agent may still call a tool.

        Args:
            budget: What the question has used so far.

        Returns:
            True while searches or reads are left.
        """
        return budget.searches < self.max_searches or budget.reads < self.max_reads

    def run_tool(
        self, tool: str, arguments: dict[str, object], question: str, budget: Budget
    ) -> tuple[dict[str, str], list[SearchHit]]:
        """Run one tool call within its budget.

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
        if tool != KEYWORD_TOOL:
            return super().run_tool(tool, arguments, question, budget)
        if budget.searches >= self.max_searches:
            return super().run_tool(tool, arguments, question, budget)  # the limit error
        budget.searches += 1

        return self.keyword_snippets(str(arguments.get("query") or question))

    def keyword_snippets(self, query: str) -> tuple[dict[str, str], list[SearchHit]]:
        """Search by keywords and show each result as its matching lines.

        Args:
            query: The keywords.

        Returns:
            The snippets as the model sees them, and the documents they come from.
        """
        found = self.knowledge_base.retrieve_keywords(query, self.keyword_results)
        texts = self.reader.read_documents([hit.doc_id for hit in found])
        if not texts:
            return {"results": "No document contains these words. Try other words."}, []

        return {"results": snippet_context(texts, query)}, texts

    def read(self, doc_id: str, budget: Budget) -> tuple[dict[str, str], list[SearchHit]]:
        """Read one document in full, within the read budget.

        Args:
            doc_id: The document's id, as a search showed it.
            budget: What the question has used so far; updated.

        Returns:
            The document's text, or an error, and the document.
        """
        if budget.reads >= self.max_reads:
            return {"error": READ_LIMIT_REACHED}, []
        hits = self.reader.read_documents([doc_id])
        if not hits:
            return {"error": f"No document has the id {doc_id}."}, []
        budget.reads += 1
        text = hits[0].text
        cut = ""
        if len(text) > READ_CHARS:
            cut = f"\n[... cut at {READ_CHARS:,} of {len(text):,} characters]"

        return {"document": f"{hits[0].source_type} ({doc_id}):\n{text[:READ_CHARS]}{cut}"}, hits
