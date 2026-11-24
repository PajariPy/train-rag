"""The field-search agent as a retriever: its final documents, measured without an answer.

For `evaluate-search`: the agent searches (general and field search) and checks, exactly as in
`field-agent`, but no answer is written and nobody is asked, so a run costs only the agent's
turns and rewrites. The ranking is the agent's final documents in the check's order, then the
rest of its candidates in reranker order (the loop's scores, so nothing is reranked twice).
With `log_to`, each question's run (searches and their results, picks, checks, the whole
conversation) is appended to a JSON Lines file, for reviewing the misses.
"""

import json
from pathlib import Path
from typing import ClassVar, Protocol

from google import genai

from train_rag.evaluation.field_agent.loop import AgentLoop
from train_rag.evaluation.field_agent.system import FieldAgentSystem
from train_rag.evaluation.hybrid import PipelineRetriever, to_hits
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.retrieval import SearchHit
from train_rag.retrieval import Retriever

# Documents kept per search in the log, best first.
LOGGED_PER_SEARCH = 50


class Searchable(Protocol):
    """The part of the general search the retriever needs for the index size."""

    def searchable_doc_ids(self) -> set[str]:
        """List the documents the search can find.

        Returns:
            Their dataset ids.
        """
        ...


class FieldAgentRetriever(Retriever):
    """Ranks documents by what the field-search agent ends with, never asking."""

    name: ClassVar[str] = "field-agent"

    def __init__(self, system: FieldAgentSystem, corpus: Searchable) -> None:
        """Keep the agent and the general search it runs on.

        Args:
            system: The field-search agent.
            corpus: The general search, for the documents it indexes.
        """
        self.system = system
        self.corpus = corpus
        self.log: Path | None = None

    def log_to(self, path: Path) -> None:
        """Append every question's run to a JSON Lines file from now on.

        Args:
            path: The file; its folder is created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log = path

    @classmethod
    def load(cls, client: genai.Client, usage: UsageMeter) -> "FieldAgentRetriever":
        """Load the agent with the accepted pipeline and the field index.

        Args:
            client: Gemini client.
            usage: Counts every Gemini call's tokens.

        Returns:
            The ready retriever.
        """
        system = FieldAgentSystem.load(client, usage)
        if not isinstance(system.retriever, PipelineRetriever):
            raise TypeError("FieldAgentSystem.load must build a PipelineRetriever")

        return cls(system, system.retriever)

    def retrieve(self, question: str, limit: int) -> list[SearchHit]:
        """Run the agent on one question and rank what it found.

        Args:
            question: The question text.
            limit: Number of documents.

        Returns:
            The final documents first, then the other candidates by reranker score.
        """
        loop = self.system.search(question, None)
        searches = loop.searches
        candidates = searches.candidates(loop.run.picks)
        reranked = to_hits(searches.retriever.rerank(question, candidates, loop.scores))

        ranking: dict[str, SearchHit] = {}
        for hit in [*loop.run.documents, *reranked]:
            ranking.setdefault(hit.doc_id, hit)

        ranked = list(ranking.values())[:limit]
        if self.log is not None:
            self.write_log(self.log, question, ranked, loop)

        return ranked

    def write_log(
        self, path: Path, question: str, ranked: list[SearchHit], loop: AgentLoop
    ) -> None:
        """Append one question's run to the log, without the documents' texts.

        Args:
            path: The log file.
            question: The question text.
            ranked: The returned ranking.
            loop: The finished agent loop.
        """
        run = loop.run.model_dump(mode="json", exclude={"documents"})
        record = {
            "question": question,
            "ranking": [hit.doc_id for hit in ranked],
            "final": [hit.doc_id for hit in loop.run.documents],
            "searches": {
                name: [hit.doc_id for hit in hits[:LOGGED_PER_SEARCH]]
                for name, hits in loop.searches.rankings.items()
            },
            "scores": loop.scores,
            "run": run,
        }
        with path.open("a") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def searchable_doc_ids(self) -> set[str]:
        """List the documents the general search indexes, which is the whole corpus.

        Returns:
            Their dataset ids.
        """
        return self.corpus.searchable_doc_ids()
