"""The person behind a question, played by Gemini, who answers the agent's questions."""

from typing import Protocol

from google import genai
from google.genai import types

from train_rag.evaluation.pipeline_answer import GEMINI_MODEL
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.memory import MemoryNote
from train_rag.prompts import prompt


class Person(Protocol):
    """Someone the agent can ask about the question they asked."""

    def reply(self, ask: str) -> str:
        """Answer one of the agent's questions.

        Args:
            ask: The agent's question.

        Returns:
            The person's reply.
        """
        ...


class SimulatedPerson:
    """Gemini playing the employee who asked, knowing only their memory note.

    It never sees the right document or the gold answer: only the question, the note and
    the exchange so far, so its replies can't hold more than the note does.
    """

    def __init__(
        self, client: genai.Client, usage: UsageMeter, question: str, note: MemoryNote
    ) -> None:
        """Set up the person for one question.

        Args:
            client: Gemini client.
            usage: Counts the tokens of the replies.
            question: The question the person asked.
            note: What they remember.
        """
        self.client = client
        self.usage = usage
        self.instructions = prompt("person_instructions", question=question, note=note.note)
        self.history: list[types.Content] = []

    def reply(self, ask: str) -> str:
        """Answer the agent, keeping the exchange for later questions.

        Args:
            ask: The agent's question.

        Returns:
            One short reply.
        """
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=ask)]))
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=self.history,
            config=types.GenerateContentConfig(
                system_instruction=self.instructions,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            ),
        )
        self.usage.add(response)
        text = (response.text or "").strip() or "I don't remember."
        self.history.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))

        return text
