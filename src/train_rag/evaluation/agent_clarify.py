"""The snippet agent that can also ask the person behind the question, a few times.

For the evaluation with clarifying questions: a question alone often can't single out the
right document among near-duplicates, and a real assistant would ask. The person is
reached through a `Person` that holds what they remember; the agent only sees replies.
"""

from dataclasses import dataclass
from typing import ClassVar

from google import genai
from google.genai import types

from train_rag.evaluation.agent import Budget
from train_rag.evaluation.agent_snippets import (
    MAX_READS,
    MAX_SEARCHES,
    SnippetAgentSystem,
    SnippetKnowledgeBase,
)
from train_rag.evaluation.agent_tools import ASK_TOOL, clarify_tools
from train_rag.evaluation.person import Person
from train_rag.evaluation.system import ClarifyingSystem
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.evaluation import SystemAnswer
from train_rag.models.retrieval import SearchHit
from train_rag.prompts import prompt

MAX_ASKS = 3
# Refused tool calls (over a budget) allowed before the agent must answer, so a model that
# keeps calling used-up tools can't loop while asks are left.
MAX_REFUSED = 2
# Searches an answered question adds, so the agent can always act on the reply.
SEARCHES_PER_REPLY = 2
ASK_LIMIT_REACHED = "No more questions: answer from what you have."
NO_ONE_TO_ASK = "There is no one to ask: answer from the documents."


@dataclass
class ClarifyBudget(Budget):
    """A question's budget, plus the person to ask and how many questions were asked."""

    person: Person | None = None
    asks: int = 0
    refused: int = 0


class ClarifyingSnippetAgent(SnippetAgentSystem, ClarifyingSystem):
    """The snippet agent with an `ask_user` tool and its own budget of questions."""

    name: ClassVar[str] = "agent-snippets-clarify"

    def __init__(
        self,
        knowledge_base: SnippetKnowledgeBase,
        client: genai.Client,
        usage: UsageMeter,
        max_asks: int = MAX_ASKS,
        max_searches: int = MAX_SEARCHES,
        max_reads: int = MAX_READS,
    ) -> None:
        """Set up the agent.

        Args:
            knowledge_base: The searches and full reads behind the tools.
            client: Gemini client.
            usage: Counts the tokens and searches.
            max_asks: Questions the agent may ask the person per question.
            max_searches: Searches allowed per question, counting the first one.
            max_reads: Documents the agent may read in full per question.
        """
        super().__init__(
            knowledge_base, client, usage, max_searches=max_searches, max_reads=max_reads
        )
        self.max_asks = max_asks
        self.instructions = prompt(
            "clarify_agent_instructions", snippet_instructions=self.instructions, max_asks=max_asks
        )

    def tools(self) -> types.Tool:
        """The snippet agent's tools and `ask_user`.

        Returns:
            The tool declarations.
        """
        return clarify_tools()

    def answer_with(self, question: str, person: Person) -> SystemAnswer:
        """Answer one question, with the person to ask.

        Args:
            question: The question text.
            person: The person who asked.

        Returns:
            The answer, the documents found and the trace, asks included.
        """
        return self.answer(question, ClarifyBudget(person=person))

    def can_call(self, budget: Budget) -> bool:
        """Whether the agent may still call a tool: searches, reads or questions left.

        Args:
            budget: What the question has used so far.

        Returns:
            True while anything is left.
        """
        if isinstance(budget, ClarifyBudget) and budget.refused >= MAX_REFUSED:
            return False
        asks_left = isinstance(budget, ClarifyBudget) and budget.asks < self.max_asks

        return asks_left or super().can_call(budget)

    def run_tool(
        self, tool: str, arguments: dict[str, object], question: str, budget: Budget
    ) -> tuple[dict[str, str], list[SearchHit]]:
        """Run one tool call; questions go to the person, within their own budget.

        Args:
            tool: Name of the called tool.
            arguments: Its arguments.
            question: The question, the query when a search call gives none.
            budget: What the question has used so far; updated.

        Returns:
            What the model receives and the documents the call showed it (none for a
            question to the person).
        """
        output, hits = self.call(tool, arguments, question, budget)
        if "error" in output and isinstance(budget, ClarifyBudget):
            budget.refused += 1

        return output, hits

    def call(
        self, tool: str, arguments: dict[str, object], question: str, budget: Budget
    ) -> tuple[dict[str, str], list[SearchHit]]:
        """Run one tool call: questions to the person here, everything else as before.

        Args:
            tool: Name of the called tool.
            arguments: Its arguments.
            question: The question, the query when a search call gives none.
            budget: What the question has used so far; updated.

        Returns:
            What the model receives and the documents the call showed it.
        """
        if tool != ASK_TOOL:
            return super().run_tool(tool, arguments, question, budget)
        if not isinstance(budget, ClarifyBudget) or budget.person is None:
            return {"error": NO_ONE_TO_ASK}, []
        if budget.asks >= self.max_asks:
            return {"error": ASK_LIMIT_REACHED}, []
        budget.asks += 1
        budget.extra_searches += SEARCHES_PER_REPLY

        return {"reply": budget.person.reply(str(arguments.get("question") or ""))}, []
