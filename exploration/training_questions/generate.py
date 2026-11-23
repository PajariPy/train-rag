"""Write one question per source document with the benchmark's prompts (DeepSeek, LangGraph).

Design: docs/brainstorm/005_synthetic_training_questions.md. Plan task 2.

Reads data/sources.json (from sources.py). Basic and miscellaneous use the benchmark's basic
prompt, semantic its semantic prompt, intra-document reasoning its intra-document prompt plus
its validation call (the question must need more than one section of the document).

One small LangGraph per document: `write` -> (intra-document only) `check` -> back to `write`
until the question passes or INTRA_ATTEMPTS are used, like the benchmark's regeneration.
Appends one line per source to data/questions.jsonl, so a rerun only does what's missing.
Stops when the spend passes MAX_SPEND.

Usage:
    uv run python exploration/training_questions/generate.py --limit 20   # pilot
    uv run python exploration/training_questions/generate.py              # everything
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from dotenv import dotenv_values
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr
from sqlalchemy import text

from train_rag.database import get_engine
from train_rag.progress.recorder import RunProgress
from train_rag.training.questions import clean_question

sys.path.insert(0, str(Path(__file__).parent))
import prompts  # noqa: E402  (the benchmark's prompts, next to this script)

MODEL = "deepseek-flash"
MAX_CHARS = 30_000
WORKERS = 8
RETRIES = 3
TEMPERATURE = 0.7  # some variety in phrasing, as the benchmark's prompts ask
MAX_SPEND = 2.00
INTRA_ATTEMPTS = 3
# Dollars per million tokens: (cache-miss input, cache-hit input, output), off-peak and peak.
PRICES = {"off-peak": (0.15, 0.003, 0.60), "peak": (0.30, 0.006, 1.20)}
PEAK_HOURS = {1, 2, 3, 6, 7, 8, 9}  # UTC, weekdays
ROOT = Path(__file__).parent.parent.parent
DATA = Path(__file__).parent / "data"
SOURCES = DATA / "sources.json"
OUTPUT = DATA / "questions.jsonl"
PROMPT_OF = {
    "basic": prompts.BASIC_QUERIES_PROMPT,
    "miscellaneous": prompts.BASIC_QUERIES_PROMPT,
    "semantic": prompts.SEMANTIC_QUERIES_PROMPT,
    "intra_document_reasoning": prompts.INTRA_DOCUMENT_REASONING_PROMPT,
}


class QuestionState(TypedDict):
    """What the graph carries for one source document."""

    question_type: str
    title: str
    body: str
    attempts: int
    question: str | None
    valid: bool


class SpendMeter:
    """Adds up DeepSeek's cost from each reply's token counts."""

    def __init__(self) -> None:
        """Start at zero."""
        self.lock = threading.Lock()
        self.spent = 0.0

    def add(self, reply: AIMessage) -> None:
        """Count one reply.

        Args:
            reply: The model's reply, with DeepSeek's token usage in its metadata.
        """
        usage = reply.response_metadata.get("token_usage", {})
        miss, hit, output = PRICES["peak" if peak_now() else "off-peak"]
        cost = (
            usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0)) * miss
            + usage.get("prompt_cache_hit_tokens", 0) * hit
            + usage.get("completion_tokens", 0) * output
        ) / 1e6
        with self.lock:
            self.spent += cost


def build_graph(key: str, meter: SpendMeter) -> CompiledStateGraph[QuestionState]:
    """Build the write -> check -> retry graph.

    Args:
        key: DeepSeek API key.
        meter: Where each reply's cost is counted.

    Returns:
        The compiled graph; invoke it with a `QuestionState`.
    """
    settings: dict[str, Any] = {
        "model": MODEL,
        "api_key": SecretStr(key),
        "max_retries": RETRIES,
        "max_tokens": 400,
        # On by default: it spends the whole token budget reasoning and returns no answer.
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    writer = ChatDeepSeek(**settings, temperature=TEMPERATURE)
    checker = ChatDeepSeek(**settings, temperature=0).bind(response_format={"type": "json_object"})

    def filled(state: QuestionState) -> dict[str, str]:
        return {"document_title": state["title"], "document_contents": state["body"][:MAX_CHARS]}

    def write(state: QuestionState) -> dict[str, Any]:
        reply = writer.invoke(PROMPT_OF[state["question_type"]].format(**filled(state)))
        meter.add(reply)
        question = clean_question(str(reply.content))

        return {
            "question": question,
            "valid": question is not None,
            "attempts": state["attempts"] + 1,
        }

    def check(state: QuestionState) -> dict[str, Any]:
        reply = checker.invoke(
            prompts.INTRA_DOCUMENT_REASONING_ANSWER_GENERATION.format(
                **filled(state), query=state["question"]
            )
        )
        meter.add(reply)
        try:
            verdict = json.loads(str(reply.content))
        except ValueError:
            verdict = {}

        return {
            "valid": bool(verdict.get("valid")) and verdict.get("gold_answer") not in (None, "N/A")
        }

    def after_write(state: QuestionState) -> str:
        if state["question_type"] != "intra_document_reasoning" or state["question"] is None:
            return END

        return "check"

    def after_check(state: QuestionState) -> str:
        return END if state["valid"] or state["attempts"] >= INTRA_ATTEMPTS else "write"

    graph = StateGraph(QuestionState)
    graph.add_node("write", write)
    graph.add_node("check", check)
    graph.add_edge(START, "write")
    graph.add_conditional_edges("write", after_write, ["check", END])
    graph.add_conditional_edges("check", after_check, ["write", END])

    return graph.compile()


def peak_now() -> bool:
    """Whether DeepSeek's peak prices apply right now.

    Returns:
        True on weekdays during DeepSeek's peak hours (UTC).
    """
    now = datetime.now(UTC)

    return now.weekday() < 5 and now.hour in PEAK_HOURS


def main() -> None:
    """Generate the missing questions, stopping at the spend limit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only this many new sources")
    args = parser.parse_args()

    sources: list[dict[str, str]] = json.loads(SOURCES.read_text())
    done = set()
    if OUTPUT.exists():
        done = {json.loads(line)["doc_id"] for line in OUTPUT.read_text().splitlines() if line}
    todo = [source for source in sources if source["doc_id"] not in done]
    if args.limit is not None:
        # a few of every type for a pilot
        by_type: dict[str, list[dict[str, str]]] = {}
        for source in todo:
            by_type.setdefault(source["question_type"], []).append(source)
        todo = [row for rows in zip(*by_type.values(), strict=False) for row in rows][: args.limit]
    with get_engine().connect() as connection:
        texts = {
            str(row.doc_id): (str(row.title), str(row.content))
            for row in connection.execute(
                text("select doc_id, title, content from documents where doc_id = any(:ids)"),
                {"ids": [source["doc_id"] for source in todo]},
            ).all()
        }
    meter = SpendMeter()
    graph = build_graph(str(dotenv_values(ROOT / ".env")["DEEPSEEK_API_KEY"]), meter)
    lock = threading.Lock()
    print(f"{len(todo)} sources to do ({len(done)} done)")

    def work(source: dict[str, str]) -> None:
        if meter.spent >= MAX_SPEND:
            return
        title, body = texts[source["doc_id"]]
        try:
            final = graph.invoke(
                {
                    "question_type": source["question_type"],
                    "title": title,
                    "body": body,
                    "attempts": 0,
                    "question": None,
                    "valid": False,
                }
            )
        except Exception as error:  # noqa: BLE001  (one failed document must not stop the run)
            print(f"failed {source['doc_id']}: {error}")
            return
        row = {**source, "question": final["question"], "valid": final["valid"]}
        with lock:
            with OUTPUT.open("a") as output:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.advance(source["question_type"], 1, note=f"${meter.spent:.3f} spent")

    with (
        RunProgress(
            get_engine(), "generate training questions (DeepSeek)", len(todo), unit="questions"
        ) as progress,
        ThreadPoolExecutor(WORKERS) as pool,
    ):
        list(pool.map(work, todo))
    print(f"spent ${meter.spent:.3f}; stopped at the limit: {meter.spent >= MAX_SPEND}")


if __name__ == "__main__":
    main()
