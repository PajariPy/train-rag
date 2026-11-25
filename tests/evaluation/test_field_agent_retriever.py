"""Tests for the field-search agent as a retriever: its ranking and that it writes no answer."""

import json
from pathlib import Path

from field_agent_fakes import FakeGemini, FakeRetriever, FakeRewriter, field_index, turn, usage
from google.genai import types

from train_rag.evaluation.field_agent.loop import FINAL_DOCUMENTS
from train_rag.evaluation.field_agent.retriever import FieldAgentRetriever
from train_rag.evaluation.field_agent.system import FieldAgentSystem
from train_rag.evaluation.registry import RETRIEVERS

QUESTION = "How did support resolve the duplicate charges?"
MINED = [f"m{n}" for n in range(1, 9)]


class FakeCorpus:
    """The general search's index: a fixed set of documents."""

    def searchable_doc_ids(self) -> set[str]:
        """List the indexed documents.

        Returns:
            Their dataset ids.
        """
        return {"m1", "g10"}


def retriever(gemini: FakeGemini, search: FakeRetriever) -> FieldAgentRetriever:
    """Build the agent retriever on fakes.

    Args:
        gemini: The fake Gemini.
        search: The fake pipeline.

    Returns:
        The retriever.
    """
    system = FieldAgentSystem(search, field_index(MINED), gemini.client(), usage())
    system.rewriter = FakeRewriter()

    return FieldAgentRetriever(system, FakeCorpus())


def agent_turns() -> list[types.GenerateContentResponse]:
    """A field search, a general search, finish, then a check that puts g10 first.

    Returns:
        The agent's turns.
    """
    return [
        turn(("search_fields", {"query": "duplicate charges", "profile": "subject"})),
        turn(("search_documents", {"query": "refund email"})),
        turn(("finish", {"ids": ["d1"]})),
        turn(("judge_cases", {"reasoning": "one", "verdict": "one_case_fits", "fits": ["d9"]})),
    ]


def test_final_documents_come_first_then_the_reranked_rest() -> None:
    """The check's order leads; the other candidates follow by score, each once."""
    gemini = FakeGemini(agent_turns(), answer="unused")
    search = FakeRetriever({"refund email": ["g10", "m2"]}, preferred=["m1", "g10", "m3"])
    agent = retriever(gemini, search)

    ranking = [hit.doc_id for hit in agent.retrieve(QUESTION, 20)]

    final, rest = ranking[:FINAL_DOCUMENTS], ranking[FINAL_DOCUMENTS:]
    assert final[0] == "g10"  # the check said it fits
    assert set(final) == {"m1", "g10", "m3", "m2", "m4"}
    assert rest == [doc_id for doc_id in MINED if doc_id not in final]
    assert len(ranking) == len(set(ranking))
    assert gemini.answer_prompts == []  # no answer is written


def test_limit_and_index_come_from_the_ranking_and_the_corpus() -> None:
    """The limit cuts the ranking; the indexed documents are the general search's."""
    gemini = FakeGemini(agent_turns(), answer="unused")
    agent = retriever(gemini, FakeRetriever({"refund email": ["g10"]}, preferred=["g10"]))

    assert len(agent.retrieve(QUESTION, 2)) == 2
    assert agent.searchable_doc_ids() == {"m1", "g10"}
    assert "field-agent" in RETRIEVERS


def test_log_keeps_each_question_run(tmp_path: Path) -> None:
    """With a log, each question appends its ranking, searches and checks as one line."""
    gemini = FakeGemini(agent_turns(), answer="unused")
    search = FakeRetriever({"refund email": ["g10", "m2"]}, preferred=["m1", "g10", "m3"])
    agent = retriever(gemini, search)
    log = tmp_path / "runs" / "agent.jsonl"
    agent.log_to(log)

    ranking = [hit.doc_id for hit in agent.retrieve(QUESTION, 20)]

    (line,) = log.read_text().splitlines()
    record = json.loads(line)
    assert record["question"] == QUESTION
    assert record["ranking"] == ranking
    assert record["final"] == ranking[:FINAL_DOCUMENTS]
    assert [found["tool"] for found in record["run"]["searches"]] == [
        "search_fields",
        "search_documents",
    ]
    assert record["run"]["checks"][0]["verdict"] == "one_case_fits"
    assert "documents" not in record["run"]
