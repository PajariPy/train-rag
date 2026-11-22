"""The agent's tools, declared for Gemini, and its system instructions.

The tools are documented like an API (what every filter value holds, how it matches, when
to use which): on the Slack dev set that was the biggest single gain (article 007).
"""

from google.genai import types

from train_rag.mined_fields.index import PROFILE_COLUMNS, FieldIndex
from train_rag.mined_fields.schema import AREA_GUIDE, AREAS, SOURCE_KINDS
from train_rag.prompts import prompt

SEARCH_DOCUMENTS = "search_documents"
SEARCH_FIELDS = "search_fields"
FINISH = "finish"
ASK_USER = "ask_user"
JUDGE_CASES = "judge_cases"
SEARCH_TOOLS = (SEARCH_DOCUMENTS, SEARCH_FIELDS)

# Results shown per search, and characters of each result's text.
SHOWN = 10
SNIPPET_CHARS = 300


def declarations(can_ask: bool) -> list[types.FunctionDeclaration]:
    """Declare every tool the agent may be offered.

    Args:
        can_ask: Include `ask_user` (there is a person to ask).

    Returns:
        The declarations; the same for every turn of a conversation.
    """
    tools = [
        types.FunctionDeclaration(
            name=SEARCH_DOCUMENTS,
            description="General search over every document of every source.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the question or key words"}
                },
                "required": ["query"],
            },
        ),
        types.FunctionDeclaration(
            name=SEARCH_FIELDS,
            description="Search the mined documents' fields by meaning, with filters.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "what to look for, a few words"},
                    "profile": {"type": "string", "enum": list(PROFILE_COLUMNS)},
                    "areas": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(AREAS)},
                        "description": "optional: keep documents with any of these areas",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(SOURCE_KINDS)},
                        "description": "optional: keep documents of these kinds",
                    },
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "optional: keep chat threads from these Slack channels",
                    },
                },
                "required": ["query", "profile"],
            },
        ),
        types.FunctionDeclaration(
            name=FINISH,
            description="Stop searching and pick the documents that answer the question.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "ids from the results, best first",
                    }
                },
                "required": ["ids"],
            },
        ),
        types.FunctionDeclaration(
            name=JUDGE_CASES,
            description="Say whether the answer depends on which final document is picked.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["same_answer", "one_case_fits", "several_cases_fit"],
                    },
                    "fits": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "ids of the documents that fit the question as asked",
                    },
                },
                "required": ["reasoning", "verdict", "fits"],
            },
        ),
    ]
    if can_ask:
        tools.append(
            types.FunctionDeclaration(
                name=ASK_USER,
                description="Ask the person who asked the question one short clarifying question.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "one short question"}
                    },
                    "required": ["question"],
                },
            )
        )

    return tools


def agent_instructions(
    index: FieldIndex,
    max_searches: int,
    max_general: int,
    max_asks: int,
    searches_per_reply: int,
) -> str:
    """Write the agent's system instructions, with the index's own counts.

    Args:
        index: The field index, for the documents per area, source and channel.
        max_searches: Searches allowed in total.
        max_general: Of those, general searches allowed.
        max_asks: Questions the agent may ask the person; 0 leaves asking out.
        searches_per_reply: Searches an answered question adds.

    Returns:
        The instructions.
    """
    areas = index.area_counts()
    sources = index.source_counts()
    channels = index.channel_counts()
    instructions = prompt(
        "field_agent_instructions",
        max_searches=max_searches,
        max_general=max_general,
        mined=f"{len(index):,}",
        mined_sources=", ".join(f"{count:,} {source}" for source, count in sources.most_common()),
        areas="\n".join(f"- {area} ({areas[area]:,}): {AREA_GUIDE[area]}" for area in AREAS),
        sources="\n".join(
            f"- {source} ({sources[source]:,}): {kind}" for source, kind in SOURCE_KINDS.items()
        ),
        channels=", ".join(f"{name} ({count:,})" for name, count in channels.most_common()),
        shown=SHOWN,
        snippet=SNIPPET_CHARS,
    )
    if max_asks:
        instructions += "\n\n" + prompt(
            "field_agent_ask_guide", max_asks=max_asks, searches_per_reply=searches_per_reply
        )

    return instructions
