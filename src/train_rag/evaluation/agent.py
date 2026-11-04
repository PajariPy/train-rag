"""A ReAct agent: Gemini answers from a first search, and can search twice more itself."""

from dataclasses import dataclass
from typing import ClassVar, Protocol

from google import genai
from google.genai import types

from train_rag.evaluation.agent_tools import (
    KEYWORD_TOOL,
    SEARCH_TOOL,
    model_step,
    search_tools,
    tool_step,
)
from train_rag.evaluation.pipeline_answer import GEMINI_MODEL, documents_context
from train_rag.evaluation.system import NOT_FOUND_ANSWER, RagSystem
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.evaluation import SystemAnswer, TraceStep
from train_rag.models.retrieval import SearchHit
from train_rag.prompts import prompt

MAX_SEARCHES = 3
LIMIT_REACHED = "Search limit reached: answer from the documents already found."


class KnowledgeBase(Protocol):
    """The two searches the agent can run."""

    def retrieve(self, question: str, limit: int) -> list[SearchHit]:
        """Search by meaning and keywords together, reranked."""
        ...

    def retrieve_keywords(self, query: str, limit: int) -> list[SearchHit]:
        """Search by keywords alone (BM25)."""
        ...


@dataclass
class Budget:
    """What one question has used so far: searches (counting the first) and full reads."""

    searches: int = 1
    reads: int = 0


class AgentSystem(RagSystem):
    """Gemini answers from a first search of the question, and can search more first.

    The question itself is searched with the full retrieval pipeline before Gemini sees
    it, so the agent starts from what the plain pipeline would give and can only add to
    it. Gemini then either answers, or calls one of two tools: the same full search with
    its own wording, or a keyword-only BM25 search for exact names and ids. After
    `max_searches` searches in total, counting the first, the tools are switched off and
    the next turn must answer. The documents reported for the answer are every document
    any search returned, in the order found.
    """

    name: ClassVar[str] = "agent-rerank"

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        client: genai.Client,
        usage: UsageMeter,
        top_k: int = 5,
        max_searches: int = MAX_SEARCHES,
    ) -> None:
        """Set up the agent.

        Args:
            knowledge_base: The searches behind the tools.
            client: Gemini client.
            usage: Counts the tokens and searches.
            top_k: Documents each search returns.
            max_searches: Searches allowed per question, counting the first one.
        """
        self.knowledge_base = knowledge_base
        self.client = client
        self.usage = usage
        self.top_k = top_k
        self.max_searches = max_searches
        self.instructions = prompt(
            "agent_instructions", max_searches=max_searches - 1, not_found=NOT_FOUND_ANSWER
        )

    def config(self, can_search: bool) -> types.GenerateContentConfig:
        """Configure one turn.

        Args:
            can_search: Whether the agent may still call a search tool.

        Returns:
            The turn's settings; without searches left, the tools are switched off.
        """
        mode = (
            types.FunctionCallingConfigMode.AUTO
            if can_search
            else types.FunctionCallingConfigMode.NONE
        )

        return types.GenerateContentConfig(
            system_instruction=self.instructions,
            tools=[self.tools()],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=mode)
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW, include_thoughts=True
            ),
        )

    def tools(self) -> types.Tool:
        """The tools Gemini may call.

        Returns:
            The general search and the keyword search.
        """
        return search_tools()

    def can_call(self, budget: Budget) -> bool:
        """Whether the agent may still call a tool.

        Args:
            budget: What the question has used so far.

        Returns:
            True while searches are left.
        """
        return budget.searches < self.max_searches

    def run_tool(
        self, tool: str, arguments: dict[str, object], question: str, budget: Budget
    ) -> tuple[dict[str, str], list[SearchHit]]:
        """Run one tool call within the budget.

        Args:
            tool: Name of the called tool.
            arguments: Its arguments.
            question: The question, the query when the call gives none.
            budget: What the question has used so far; updated.

        Returns:
            What the model receives (the documents' text, or an error) and the documents
            the call found.
        """
        if budget.searches >= self.max_searches:
            return {"error": LIMIT_REACHED}, []
        budget.searches += 1
        hits = self.search(tool, str(arguments.get("query") or question))

        return {"documents": documents_context(hits)}, hits

    def search(self, tool: str, query: str) -> list[SearchHit]:
        """Run the search a tool call asks for.

        Args:
            tool: Name of the called tool.
            query: Its query.

        Returns:
            The top documents.
        """
        if tool == KEYWORD_TOOL:
            return self.knowledge_base.retrieve_keywords(query, self.top_k)

        return self.knowledge_base.retrieve(query, self.top_k)

    def answer(self, question: str) -> SystemAnswer:
        """Search the question, then let the agent search more or answer.

        Every step is recorded in the answer's trace: the instructions, the first message,
        each model turn with its thoughts and tool calls, and each search with its query
        and the documents it returned.

        Args:
            question: The question text.

        Returns:
            The answer, every document the searches returned in the order found, and the
            trace.
        """
        hits = self.knowledge_base.retrieve(question, self.top_k)
        found = [hit.doc_id for hit in hits]
        budget = Budget()
        documents = documents_context(hits)
        first_turn = prompt("agent_first_turn", question=question, documents=documents)
        trace = [
            TraceStep(kind="instructions", text=self.instructions),
            tool_step(SEARCH_TOOL, {"query": question, "automatic": True}, hits, documents),
            TraceStep(kind="input", text=first_turn),
        ]
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=first_turn)])]
        while True:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=self.config(can_search=self.can_call(budget)),
            )
            self.usage.add(response)
            trace.append(model_step(response))
            calls = response.function_calls or []
            candidate = response.candidates[0].content if response.candidates else None
            if not calls or candidate is None:
                self.usage.add_searches(budget.searches)
                return SystemAnswer(
                    answer=trace[-1].text.strip() or NOT_FOUND_ANSWER,
                    retrieved_doc_ids=list(dict.fromkeys(found)),
                    trace=trace,
                )

            # Keep the model's turn as is: it carries the thought signatures Gemini needs.
            contents.append(candidate)
            results = []
            for call in calls:
                tool = call.name or SEARCH_TOOL
                arguments = dict(call.args or {})
                output, hits = self.run_tool(tool, arguments, question, budget)
                found.extend(hit.doc_id for hit in hits)
                trace.append(tool_step(tool, arguments, hits, next(iter(output.values()))))
                results.append(types.Part.from_function_response(name=tool, response=output))
            contents.append(types.Content(role="user", parts=results))
