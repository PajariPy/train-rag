"""The language model behind the search agent: Gemini (default) or DeepSeek.

Both backends hold one conversation and expose the same three things to agent_search.py:
`step(tools)` sends the conversation and returns the tool calls the model made, `reply(call,
text)` adds a tool result, and `usage` counts tokens, priced by `cost()`. Tools are written
once, in the OpenAI function format; the Gemini backend converts them.

Gemini is told to always call a tool (search, finish or ask), so a run never ends on plain
text. DeepSeek keeps its earlier setup (no thinking, temperature 0).
"""

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from google import genai
from google.genai import types
from label import MODEL as DEEPSEEK_MODEL
from label import PRICES

from train_rag.evaluation.judge import GEMINI_MODEL
from train_rag.evaluation.usage import CACHED_INPUT_PRICE, INPUT_PRICE, OUTPUT_PRICE

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


@dataclass
class Call:
    """One tool call: its id, the tool's name and the parsed arguments."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class DeepSeekChat:
    """A DeepSeek conversation over the OpenAI-style chat API."""

    client: httpx.Client
    system: str
    user: str
    usage: dict[str, int] = field(default_factory=lambda: {"miss": 0, "hit": 0, "output": 0})
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Start with the system and user messages."""
        self.messages = [
            {"role": "system", "content": self.system}, {"role": "user", "content": self.user}
        ]

    def step(self, tools: list[dict[str, Any]]) -> list[Call]:
        """Send the conversation; the model's tool calls."""
        payload = {
            "model": DEEPSEEK_MODEL, "messages": self.messages, "tools": tools,
            "temperature": 0, "thinking": {"type": "disabled"}, "max_tokens": 500,
        }
        response = self.client.post(DEEPSEEK_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        used = data.get("usage", {})
        self.usage["hit"] += used.get("prompt_cache_hit_tokens", 0)
        self.usage["miss"] += used.get("prompt_cache_miss_tokens", used.get("prompt_tokens", 0))
        self.usage["output"] += used.get("completion_tokens", 0)
        message = data["choices"][0]["message"]
        self.messages.append(
            {k: v for k, v in message.items() if k in ("role", "content", "tool_calls")}
        )
        calls = []
        for call in message.get("tool_calls") or []:
            try:
                args = json.loads(call["function"]["arguments"])
            except ValueError as error:
                args = {"_error": str(error)}
            calls.append(Call(call["id"], call["function"]["name"], args))
        return calls

    def reply(self, call: Call, text: str) -> None:
        """Add one tool result."""
        self.messages.append({"role": "tool", "tool_call_id": call.id, "content": text})

    def cost(self) -> float:
        """Dollars at off-peak prices."""
        miss, hit, output = PRICES["off-peak"]
        u = self.usage
        return (u["miss"] * miss + u["hit"] * hit + u["output"] * output) / 1e6


@dataclass
class GeminiChat:
    """A Gemini conversation with function calling."""

    client: genai.Client
    system: str
    user: str
    usage: dict[str, int] = field(
        default_factory=lambda: {"input": 0, "cached": 0, "output": 0, "thinking": 0}
    )
    contents: list[types.Content] = field(default_factory=list)
    pending: list[types.Part] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Start with the user message (the system prompt goes in the config)."""
        self.contents = [types.Content(role="user", parts=[types.Part(text=self.user)])]

    def step(self, tools: list[dict[str, Any]]) -> list[Call]:
        """Send the conversation (with the pending tool results); the model's tool calls."""
        if self.pending:
            self.contents.append(types.Content(role="user", parts=self.pending))
            self.pending = []
        declarations = [
            types.FunctionDeclaration(
                name=t["function"]["name"], description=t["function"]["description"],
                parameters_json_schema=t["function"]["parameters"],
            )
            for t in tools
        ]
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=self.contents,
            config=types.GenerateContentConfig(
                system_instruction=self.system,
                tools=[types.Tool(function_declarations=declarations)],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            ),
        )
        if (used := response.usage_metadata) is not None:
            cached = used.cached_content_token_count or 0
            self.usage["input"] += (used.prompt_token_count or 0) - cached
            self.usage["cached"] += cached
            self.usage["output"] += used.candidates_token_count or 0
            self.usage["thinking"] += used.thoughts_token_count or 0
        if not response.candidates or response.candidates[0].content is None:
            raise ValueError("Gemini returned no candidate")
        # Keep the model's turn as it came: it carries the thought signatures.
        self.contents.append(response.candidates[0].content)
        return [
            Call(f.id or f"call{i}", f.name or "", dict(f.args or {}))
            for i, f in enumerate(response.function_calls or [])
        ]

    def reply(self, call: Call, text: str) -> None:
        """Queue one tool result; all results of a turn go back together."""
        self.pending.append(types.Part(function_response=types.FunctionResponse(
            id=call.id if not call.id.startswith("call") else None,
            name=call.name, response={"result": text},
        )))

    def cost(self) -> float:
        """Dollars at Gemini Flash standard prices (thinking billed as output)."""
        u = self.usage
        return float(
            u["input"] * INPUT_PRICE + u["cached"] * CACHED_INPUT_PRICE
            + (u["output"] + u["thinking"]) * OUTPUT_PRICE
        ) / 1e6
