"""One Gemini conversation with function calling, for the field-search agent.

Moved from exploration/slack_extraction/agent_llm.py (the DeepSeek backend stays there).
Every turn declares the same tools, so the conversation's start stays one prefix that
Gemini's implicit cache can reuse; which tools the model may call in a turn is set with
`allowed_function_names`, and it must call one (mode ANY), so a run never ends on text.
"""

from google import genai
from google.genai import types

from train_rag.evaluation.pipeline_answer import GEMINI_MODEL
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.evaluation import TraceStep
from train_rag.models.field_agent import ToolCall


class GeminiChat:
    """A conversation: the system instructions, the turns so far and the tools."""

    def __init__(
        self,
        client: genai.Client,
        usage: UsageMeter,
        instructions: str,
        user: str,
        tools: list[types.FunctionDeclaration],
    ) -> None:
        """Start the conversation with the user's message.

        Args:
            client: Gemini client.
            usage: Counts the tokens of every turn.
            instructions: The system instructions.
            user: The first message.
            tools: Every tool the model may be offered.
        """
        self.client = client
        self.usage = usage
        self.instructions = instructions
        self.tools = types.Tool(function_declarations=tools)
        self.contents = [types.Content(role="user", parts=[types.Part(text=user)])]
        self.pending: list[types.Part] = []
        self.trace = [
            TraceStep(kind="instructions", text=instructions),
            TraceStep(kind="input", text=user),
        ]

    def step(self, allowed: list[str]) -> list[ToolCall]:
        """Send the conversation with the pending tool results and get the model's calls.

        Args:
            allowed: Names of the tools the model may call in this turn.

        Returns:
            The tool calls the model made.

        Raises:
            ValueError: If Gemini returns no candidate.
        """
        if self.pending:
            self.contents.append(types.Content(role="user", parts=self.pending))
            self.pending = []
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=self.contents,
            config=types.GenerateContentConfig(
                system_instruction=self.instructions,
                tools=[self.tools],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=allowed,
                    )
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            ),
        )
        self.usage.add(response)
        if not response.candidates or response.candidates[0].content is None:
            raise ValueError("Gemini returned no candidate")

        # Keep the model's turn as it came: it carries the thought signatures.
        self.contents.append(response.candidates[0].content)
        calls = [
            ToolCall(id=call.id, name=call.name or "", args=dict(call.args or {}))
            for call in response.function_calls or []
        ]
        used = response.usage_metadata
        self.trace.append(
            TraceStep(
                kind="model",
                text=response.text or "" if not calls else "",
                tool_calls=[{"tool": call.name, "arguments": call.args} for call in calls],
                input_tokens=(used.prompt_token_count or 0) if used else 0,
                output_tokens=(used.candidates_token_count or 0) if used else 0,
                thinking_tokens=(used.thoughts_token_count or 0) if used else 0,
            )
        )

        return calls

    def reply(self, call: ToolCall, text: str, doc_ids: list[str] | None = None) -> None:
        """Queue one tool result; all results of a turn go back together.

        Args:
            call: The call answered.
            text: What the model receives.
            doc_ids: The documents shown in it, for the trace.
        """
        response = types.FunctionResponse(id=call.id, name=call.name, response={"result": text})
        self.pending.append(types.Part(function_response=response))
        self.trace.append(
            TraceStep(
                kind="tool",
                tool=call.name,
                arguments=call.args,
                output=text,
                doc_ids=doc_ids or [],
            )
        )
