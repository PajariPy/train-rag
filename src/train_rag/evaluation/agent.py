"""A ReAct agent: Gemini answers from a first search, and can search twice more itself."""

from typing import ClassVar, Protocol

from google import genai
from google.genai import types

from train_rag.evaluation.pipeline_answer import GEMINI_MODEL, documents_context
from train_rag.evaluation.system import NOT_FOUND_ANSWER, RagSystem
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.evaluation import SystemAnswer, TraceStep
from train_rag.models.retrieval import SearchHit
from train_rag.prompts import prompt

MAX_SEARCHES = 3
SEARCH_TOOL = "search_knowledge_base"
KEYWORD_TOOL = "keyword_search"
LIMIT_REACHED = "Search limit reached: answer from the documents already found."


class KnowledgeBase(Protocol):
    """The two searches the agent can run."""

    def retrieve(self, question: str, limit: int) -> list[SearchHit]:
        """Search by meaning and keywords together, reranked."""
        ...

    def retrieve_keywords(self, query: str, limit: int) -> list[SearchHit]:
        """Search by keywords alone (BM25)."""
        ...


def search_function(
    name: str, description: str, query_description: str
) -> types.FunctionDeclaration:
    """Describe one search tool that takes a single query.

    Args:
        name: Function name the model calls.
        description: What the search does and when to use it.
        query_description: How to write the query.

    Returns:
        The function declaration.
    """
    return types.FunctionDeclaration(
        name=name,
        description=description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(type=types.Type.STRING, description=query_description)
            },
            required=["query"],
        ),
    )


def search_tools() -> types.Tool:
    """Describe the agent's two searches to Gemini.

    Returns:
        The tool with the general search and the keyword search.
    """
    return types.Tool(
        function_declarations=[
            search_function(
                SEARCH_TOOL,
                prompt("search_tool_description"),
                prompt("search_query_description"),
            ),
            search_function(
                KEYWORD_TOOL,
                prompt("keyword_tool_description"),
                prompt("keyword_query_description"),
            ),
        ]
    )


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
            tools=[search_tools()],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=mode)
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW, include_thoughts=True
            ),
        )

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
        searches = 1
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
                config=self.config(can_search=searches < self.max_searches),
            )
            self.usage.add(response)
            trace.append(model_step(response))
            calls = response.function_calls or []
            candidate = response.candidates[0].content if response.candidates else None
            if not calls or candidate is None:
                self.usage.add_searches(searches)
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
                output: dict[str, str] = {"error": LIMIT_REACHED}
                hits = []
                if searches < self.max_searches:
                    searches += 1
                    hits = self.search(tool, str(arguments.get("query") or question))
                    found.extend(hit.doc_id for hit in hits)
                    output = {"documents": documents_context(hits)}
                trace.append(tool_step(tool, arguments, hits, next(iter(output.values()))))
                results.append(types.Part.from_function_response(name=tool, response=output))
            contents.append(types.Content(role="user", parts=results))


def tool_step(
    tool: str, arguments: dict[str, object], hits: list[SearchHit], output: str
) -> TraceStep:
    """Record one search.

    Args:
        tool: The search tool.
        arguments: Its arguments.
        hits: The documents it returned.
        output: What the model received: the documents' text, or an error.

    Returns:
        The trace step.
    """
    return TraceStep(
        kind="tool",
        tool=tool,
        arguments=arguments,
        output=output,
        doc_ids=[hit.doc_id for hit in hits],
    )


def model_step(response: types.GenerateContentResponse) -> TraceStep:
    """Record one model turn: its text, thought summary, tool calls and tokens.

    Args:
        response: Gemini's response.

    Returns:
        The trace step.
    """
    content = response.candidates[0].content if response.candidates else None
    parts = (content.parts if content else None) or []
    usage = response.usage_metadata

    return TraceStep(
        kind="model",
        text="".join(part.text or "" for part in parts if not part.thought),
        thoughts="".join(part.text or "" for part in parts if part.thought),
        tool_calls=[
            {"tool": call.name, "arguments": dict(call.args or {})}
            for call in response.function_calls or []
        ],
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        thinking_tokens=(usage.thoughts_token_count or 0) if usage else 0,
    )
