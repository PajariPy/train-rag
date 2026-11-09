"""Rerank on the first N tokens of each document: the speed of fact blocks with no extraction?

Fact blocks made the reranker about 5× faster (11 instead of 59 ms per document) with the
same top 5 on the candidates. This checks whether simply cutting each document to its first
N tokens (the title comes first) gives the same, with no model and nothing to keep up to date.

Two settings, the 46 semantic failures:

1. The candidates (gte top 50 + the expected document), against the cached full-text and
   fact-block scores of rerank_pilot.py.
2. The real 300-deep fused lists of e2e_pilot.py, against its full-text scores.

Scores are cached in data/trunc_pilot_scores.json; writes data/trunc_pilot.json.
Free: no API calls.

    uv run python exploration/fact_extraction/trunc_pilot.py
"""

import json
import time
from pathlib import Path

from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.processing.normalize import normalize_documents, normalize_text
from train_rag.progress.recorder import RunProgress
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.reranker import GteModernBertReranker

HERE = Path(__file__).parent
POOLS = HERE.parent / "retrieval_failures" / "data" / "pools_run15.json"
PILOT_SCORES = HERE / "data" / "rerank_pilot_scores.json"
E2E_SCORES = HERE / "data" / "e2e_pilot_scores.json"
SCORES = HERE / "data" / "trunc_pilot_scores.json"
OUT = HERE / "data" / "trunc_pilot.json"
LIMITS = [256, 512, 1024]


def load_texts(doc_ids: list[str]) -> dict[str, str]:
    """Full normalized title + text of documents, keyed by dataset id."""
    with Session(get_engine()) as session:
        internal = ids_of_doc_ids(session, EnterpriseRagBenchLoader.name, doc_ids)
        clean = normalize_documents(load_documents(session, sorted(internal)))

    return {doc.doc_id: f"{doc.title}\n\n{doc.text}" for doc in clean}


def score_all(pools: dict, pilot: dict, e2e: dict) -> dict:
    """Rerank every question's documents cut to each limit, in both settings; cached."""
    cache: dict = json.loads(SCORES.read_text()) if SCORES.exists() else {}
    todo = [(setting, qid) for setting in ("candidates", "e2e") for qid in pools]
    todo = [(s, q) for s, q in todo if q not in cache.get(s, {})]
    if not todo:
        return cache
    reranker = GteModernBertReranker(batch_size=4)
    tokenizer = reranker.model.tokenizer
    documents = json.loads(POOLS.read_text())["documents"]

    def cut(text: str, limit: int) -> str:
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        return text if len(ids) <= limit else tokenizer.decode(ids[:limit])

    with RunProgress(get_engine(), "trunc pilot: rerank first N tokens", len(todo)) as progress:
        for setting, qid in todo:
            question = normalize_text(pools[qid]["question"])
            if setting == "candidates":
                ids = list(pilot[qid]["rerank_text"])
                texts = {d: documents[d] for d in ids}
            else:
                ids = e2e[qid]["fused"]
                texts = load_texts(ids)
            entry: dict = {"scores": {}, "seconds": {}}
            for limit in LIMITS:
                short = [cut(texts[d], limit) for d in ids]
                started = time.perf_counter()
                entry["scores"][str(limit)] = dict(
                    zip(ids, reranker.score(question, short), strict=True)
                )
                entry["seconds"][str(limit)] = time.perf_counter() - started
            cache.setdefault(setting, {})[qid] = entry
            SCORES.write_text(json.dumps(cache))
            progress.advance("question-settings", 1, note=f"{setting} {qid}")

    return cache


def rank_of(scores: dict[str, float], ids: list[str], expected: str) -> int | None:
    """1-based rank of the expected document among `ids` by score, or None if absent."""
    if expected not in ids:
        return None

    return sorted(ids, key=lambda d: -scores[d]).index(expected) + 1


def report(name: str, ranks: list[int | None], ms: float | None) -> dict:
    """Print one table line and return its numbers."""
    n = len(ranks)
    top = {k: sum(r is not None and r <= k for r in ranks) for k in (1, 5, 10)}
    timing = f"{ms:>9.1f}" if ms is not None else f"{'-':>9}"
    print(
        f"{name:34} {top[1]:>6} {top[5]:>6} {top[10]:>7} {100 * top[5] / n:>8.0f}% "
        f"{100 * top[10] / n:>9.0f}% {n - top[10]:>7} {timing}"
    )

    return {"top1": top[1], "top5": top[5], "top10": top[10], "ms_per_doc": ms}


def main() -> None:
    """Score, then compare the cut texts with full text and fact blocks."""
    pools = json.loads(POOLS.read_text())["pools"]
    pilot = json.loads(PILOT_SCORES.read_text())
    timing = pilot.pop("_timing")
    e2e = json.loads(E2E_SCORES.read_text())
    cache = score_all(pools, pilot, e2e)
    header = (
        f"{'variant':34} {'top 1':>6} {'top 5':>6} {'top 10':>7} {'recall@5':>9} "
        f"{'recall@10':>10} {'missed':>7} {'ms/doc':>9}"
    )
    results: dict = {}

    print("\n1. Candidates (~53 per question, right document always present)")
    print(header)
    part: dict = {}
    for name, key, ms in [
        ("full text", "rerank_text", timing["ms_per_doc_text"]),
        ("fact block", "rerank_facts", timing["ms_per_doc_facts"]),
    ]:
        ranks = [rank_of(pilot[q][key], list(pilot[q][key]), pools[q]["expected"]) for q in pools]
        part[name] = report(name, ranks, ms)
    for limit in LIMITS:
        key = str(limit)
        docs = sum(len(pilot[q]["rerank_text"]) for q in pools)
        ms = 1000 * sum(cache["candidates"][q]["seconds"][key] for q in pools) / docs
        ranks = [
            rank_of(
                cache["candidates"][q]["scores"][key],
                list(pilot[q]["rerank_text"]),
                pools[q]["expected"],
            )
            for q in pools
        ]
        part[f"first {limit} tokens"] = report(f"first {limit} tokens", ranks, ms)
    results["candidates"] = part

    print("\n2. End to end (today's hybrid search 300 deep, then rerank all 300)")
    print(header)
    part = {}
    docs = sum(len(e2e[q]["fused"]) for q in pools)
    ms = 1000 * sum(e2e[q]["seconds"]["rerank_text"] for q in pools) / docs
    ranks = [rank_of(e2e[q]["rerank_text"], e2e[q]["fused"], pools[q]["expected"]) for q in pools]
    part["full text"] = report("full text", ranks, ms)
    for limit in LIMITS:
        key = str(limit)
        ms = 1000 * sum(cache["e2e"][q]["seconds"][key] for q in pools) / docs
        ranks = [
            rank_of(cache["e2e"][q]["scores"][key], e2e[q]["fused"], pools[q]["expected"])
            for q in pools
        ]
        part[f"first {limit} tokens"] = report(f"first {limit} tokens", ranks, ms)
    results["e2e"] = part
    OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT}")


main()
