"""The agent's tools as Gemini sees them, and the trace steps recorded for each turn."""

from google.genai import types

from train_rag.models.evaluation import TraceStep
from train_rag.models.retrieval import SearchHit
from train_rag.prompts import prompt

SEARCH_TOOL = "search_knowledge_base"
KEYWORD_TOOL = "keyword_search"
READ_TOOL = "read_document"
ASK_TOOL = "ask_user"


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


def snippet_tools() -> types.Tool:
    """Describe the snippet agent's tools: the general search, keyword snippets, full reads.

    Returns:
        The general search, the keyword search that returns snippets, and the tool that
        reads one document in full by its id.
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
                prompt("keyword_snippet_tool_description"),
                prompt("keyword_query_description"),
            ),
            types.FunctionDeclaration(
                name=READ_TOOL,
                description=prompt("read_tool_description"),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "doc_id": types.Schema(
                            type=types.Type.STRING, description=prompt("read_id_description")
                        )
                    },
                    required=["doc_id"],
                ),
            ),
        ]
    )


def clarify_tools() -> types.Tool:
    """Describe the clarifying agent's tools: the snippet agent's, plus asking the person.

    Returns:
        The snippet agent's three tools and `ask_user`, which takes one question.
    """
    return types.Tool(
        function_declarations=[
            *(snippet_tools().function_declarations or []),
            types.FunctionDeclaration(
                name=ASK_TOOL,
                description=prompt("ask_tool_description"),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "question": types.Schema(
                            type=types.Type.STRING, description=prompt("ask_question_description")
                        )
                    },
                    required=["question"],
                ),
            ),
        ]
    )


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
