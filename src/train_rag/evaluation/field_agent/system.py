"""The field-search agent as an evaluated system: search, check, then answer with citations.

`field-agent` never asks; `field-agent-clarify` may ask the simulated person twice. Both
answer with `document_answer_instructions` (the main case plus similar cases, cited inline),
and show the cases side by side when several still fit at the end.
"""

from typing import ClassVar

from google import genai

from train_rag.database import get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.evaluation.field_agent.chat import GeminiChat
from train_rag.evaluation.field_agent.loop import SEARCHES_PER_REPLY, AgentLoop
from train_rag.evaluation.field_agent.searches import (
    MAX_GENERAL_SEARCHES,
    MAX_SEARCHES,
    AgentSearches,
    DocumentSearch,
)
from train_rag.evaluation.field_agent.tools import agent_instructions, declarations
from train_rag.evaluation.hybrid import PipelineRetriever
from train_rag.evaluation.person import Person
from train_rag.evaluation.pipeline_answer import answer_from_hits
from train_rag.evaluation.system import ClarifyingSystem
from train_rag.evaluation.usage import UsageMeter
from train_rag.mined_fields.index import FieldSearch
from train_rag.mined_fields.stored import StoredFieldIndex
from train_rag.models.evaluation import SystemAnswer
from train_rag.pipelines.retrieval import RetrievalPipeline
from train_rag.prompts import prompt
from train_rag.query_rewrite import QueryRewriter
from train_rag.reranker import SHORT_TOKENS

MAX_ASKS = 2


class FieldAgentSystem(ClarifyingSystem):
    """The agent with the general and the field search; it never asks."""

    name: ClassVar[str] = "field-agent"
    max_asks: ClassVar[int] = 0

    def __init__(
        self,
        retriever: DocumentSearch,
        index: FieldSearch,
        client: genai.Client,
        usage: UsageMeter,
    ) -> None:
        """Set up the system.

        Args:
            retriever: The general search, reranker and document reader.
            index: The mined fields.
            client: Gemini client for the agent, the rewrites and the answer.
            usage: Counts every Gemini call's tokens.
        """
        self.retriever = retriever
        self.index = index
        self.client = client
        self.usage = usage
        self.rewriter = QueryRewriter(client, usage)
        self.instructions = {
            asks: agent_instructions(
                index, MAX_SEARCHES, MAX_GENERAL_SEARCHES, asks, SEARCHES_PER_REPLY
            )
            for asks in {0, self.max_asks}
        }

    @classmethod
    def load(cls, client: genai.Client, usage: UsageMeter) -> "FieldAgentSystem":
        """Load the pipeline (union rerank on 1,024 tokens) and field search (in Postgres).

        Both search with the same fine-tuned gte.

        Args:
            client: Gemini client.
            usage: Counts every Gemini call's tokens.

        Returns:
            The ready system.
        """
        embedder = FineTunedGteEmbedder()
        pipeline = RetrievalPipeline.load(embedder=embedder, union=True, rerank_tokens=SHORT_TOKENS)

        return cls(
            PipelineRetriever(pipeline), StoredFieldIndex(get_engine(), embedder), client, usage
        )

    def answer(self, question: str) -> SystemAnswer:
        """Answer one question with no one to ask.

        Args:
            question: The question text.

        Returns:
            The answer, the final documents and the agent's trace.
        """
        return self.solve(question, None)

    def answer_with(self, question: str, person: Person) -> SystemAnswer:
        """Answer one question, asking the person when this system may.

        Args:
            question: The question text.
            person: The person who asked.

        Returns:
            The answer, the final documents and the agent's trace.
        """
        return self.solve(question, person if self.max_asks else None)

    def solve(self, question: str, person: Person | None) -> SystemAnswer:
        """Let the agent search and check, then answer from its final documents.

        Args:
            question: The question text.
            person: Who to ask, or None.

        Returns:
            The answer, the final documents and the agent's trace.
        """
        run = self.search(question, person).run
        note = prompt("side_by_side_note") if run.side_by_side else ""
        answer = answer_from_hits(self.client, self.usage, question, run.documents, note)

        return answer.model_copy(update={"trace": run.trace})

    def search(self, question: str, person: Person | None) -> AgentLoop:
        """Let the agent search and check, without writing an answer.

        Args:
            question: The question text.
            person: Who to ask, or None.

        Returns:
            The finished loop: its run (final documents, trace), searches and rerank scores.
        """
        asks = self.max_asks if person is not None else 0
        chat = GeminiChat(
            self.client,
            self.usage,
            self.instructions[asks],
            f"Search question: {question}",
            declarations(can_ask=asks > 0),
        )

        searches = AgentSearches(self.retriever, self.index, self.rewriter)
        loop = AgentLoop(chat, searches, question, person, asks)
        loop.execute()
        self.usage.add_searches(len(loop.run.searches))

        return loop


class ClarifyingFieldAgent(FieldAgentSystem):
    """The same agent, asking the person at most `MAX_ASKS` times."""

    name: ClassVar[str] = "field-agent-clarify"
    max_asks: ClassVar[int] = MAX_ASKS
