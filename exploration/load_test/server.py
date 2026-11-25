"""Stand-in for the query service, for a load test on one core and 2 GB.

BM25 is real: the full index (500k documents) answers each question on the server's
core, called through a thread like `pipelines/retrieval.py` does. Every other service is
a wait of a realistic length: the work happens elsewhere (SageMaker, Postgres, the LLM),
the query service only waits for it. The answer streams back as server-sent events.

Settings come from environment variables (seconds unless noted):
    BM25_DIR=/index  BM25_MMAP=0|1  LOAD_DOCUMENTS=1|0
    EMBED_S=0.05 PG_S=0.05 RERANK_S=0.3 AGENT_S=1.5 FIRST_TOKEN_S=0.8
    CHUNKS=30 CHUNK_S=0.1 CONTEXT_DOCS=100 DOC_CHARS=3000
"""

import asyncio
import json
import os
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

import bm25s
import Stemmer
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


def setting(name: str, default: float) -> float:
    """Read a number from the environment."""
    return float(os.environ.get(name, default))


INDEX = Path(os.environ.get("BM25_DIR", "/index"))
MMAP = os.environ.get("BM25_MMAP", "0") == "1"
LOAD_DOCUMENTS = os.environ.get("LOAD_DOCUMENTS", "1") == "1"
EMBED_S = setting("EMBED_S", 0.05)  # SageMaker embedder endpoint
PG_S = setting("PG_S", 0.05)  # vector search in Postgres
RERANK_S = setting("RERANK_S", 0.3)  # SageMaker reranker endpoint, ~100 documents
AGENT_S = setting("AGENT_S", 1.5)  # field agent: one LLM call
FIRST_TOKEN_S = setting("FIRST_TOKEN_S", 0.8)  # answer LLM: time to first token
CHUNKS = int(setting("CHUNKS", 30))  # streamed answer chunks
CHUNK_S = setting("CHUNK_S", 0.1)  # time between chunks
CONTEXT_DOCS = int(setting("CONTEXT_DOCS", 100))  # documents held per question
DOC_CHARS = int(setting("DOC_CHARS", 3000))
CANDIDATES = 50  # BM25 candidates, as in pipelines/retrieval.py

started = time.perf_counter()
model = bm25s.BM25.load(INDEX, mmap=MMAP)
# The real retriever keeps every document's ids in memory; keep the same load.
documents = json.loads((INDEX / "documents.json").read_text()) if LOAD_DOCUMENTS else []
stemmer = Stemmer.Stemmer("english")
lock = threading.Lock()
LOAD_SECONDS = time.perf_counter() - started

stats: dict[str, float] = {"bm25_n": 0, "bm25_s": 0.0, "bm25_max": 0.0, "lag_max": 0.0, "lag_sum": 0.0, "lag_n": 0}
in_flight = 0

app = FastAPI()


class Question(BaseModel):
    """One question."""

    question: str


def bm25(question: str) -> list[int]:
    """Run the real BM25 search; return the candidate positions."""
    t = time.perf_counter()
    with lock:
        tokens = bm25s.tokenize(
            [question], stopwords="en", stemmer=stemmer.stemWords, return_ids=False, show_progress=False
        )
        indexes, _ = model.retrieve([list(tokens[0])], k=CANDIDATES, show_progress=False)
    took = time.perf_counter() - t
    stats["bm25_n"] += 1
    stats["bm25_s"] += took
    stats["bm25_max"] = max(stats["bm25_max"], took)
    return indexes[0].tolist()


async def answer(question: str) -> AsyncIterator[str]:
    """Go through the query path and stream the answer."""
    global in_flight
    in_flight += 1
    try:
        await asyncio.sleep(EMBED_S)
        candidates = await asyncio.to_thread(bm25, question)
        await asyncio.sleep(PG_S)
        context = [f"{i} {'x' * DOC_CHARS}" for i in range(CONTEXT_DOCS)]  # fetched document texts
        await asyncio.sleep(RERANK_S)
        await asyncio.sleep(AGENT_S)
        prompt = question + "\n".join(context)
        await asyncio.sleep(FIRST_TOKEN_S)
        for i in range(CHUNKS):
            yield f"data: {json.dumps({'chunk': i, 'text': 'some answer words ' * 4})}\n\n"
            await asyncio.sleep(CHUNK_S)
        yield f"data: {json.dumps({'done': True, 'candidates': len(candidates), 'prompt': len(prompt)})}\n\n"
    finally:
        in_flight -= 1


@app.post("/ask")
async def ask(body: Question) -> StreamingResponse:
    """Answer one question as a server-sent event stream."""
    return StreamingResponse(answer(body.question), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict[str, object]:
    """Report that the index is loaded."""
    return {"ok": True, "load_seconds": round(LOAD_SECONDS, 1), "mmap": MMAP, "documents": len(documents)}


@app.get("/stats")
async def read_stats(reset: bool = True) -> dict[str, float]:
    """BM25 time per question and event-loop lag since the last reset."""
    n, lag_n = stats["bm25_n"] or 1, stats["lag_n"] or 1
    out = {
        "bm25_avg_ms": 1000 * stats["bm25_s"] / n,
        "bm25_max_ms": 1000 * stats["bm25_max"],
        "loop_lag_avg_ms": 1000 * stats["lag_sum"] / lag_n,
        "loop_lag_max_ms": 1000 * stats["lag_max"],
        "in_flight": in_flight,
    }
    if reset:
        stats.update(bm25_n=0, bm25_s=0.0, bm25_max=0.0, lag_max=0.0, lag_sum=0.0, lag_n=0)
    return out


@app.on_event("startup")
async def watch_loop() -> None:
    """Measure event-loop lag: how late a 100 ms timer fires."""

    async def watch() -> None:
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.1)
            lag = time.perf_counter() - t - 0.1
            stats["lag_max"] = max(stats["lag_max"], lag)
            stats["lag_sum"] += lag
            stats["lag_n"] += 1

    asyncio.get_running_loop().create_task(watch())
