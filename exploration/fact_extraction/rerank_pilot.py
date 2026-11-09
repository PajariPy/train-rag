"""Reranker + BM25 over facts on the candidates of the 46 semantic failures.

Every failure has ~51 candidates (gte's top 50 plus the expected document). The reranker
scores each candidate twice: on the document text (up to its 8,192 tokens) and on the
DeepSeek fact block only. The scores are fused with the keyword rankings cached by
retrieval_pilot.py, and two-stage variants first keep the top k of a cheap fusion and
rerank only those.

Writes data/rerank_pilot.json (ranks) and caches the reranker scores in
data/rerank_pilot_scores.json. Free: no API calls.

    uv run python exploration/fact_extraction/rerank_pilot.py
"""

import json
import time
from pathlib import Path

from train_rag.database import get_engine
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress
from train_rag.reranker import GteModernBertReranker

HERE = Path(__file__).parent
POOLS = HERE.parent / "retrieval_failures" / "data" / "pools_run15.json"
LABELS = HERE / "data" / "labels.json"
BASE_SCORES = HERE / "data" / "retrieval_pilot_scores.json"
SCORES = HERE / "data" / "rerank_pilot_scores.json"
OUT = HERE / "data" / "rerank_pilot.json"
RRF_K = 60
WEIGHTS = [0.25, 0.5, 1.0, 2.0]
FIRST_STAGE = ["bm25_text", "bm25_facts", "gte_text"]  # cheap fusion, gte at 0.25
FIRST_WEIGHTS = [1.0, 1.0, 0.25]
FUSIONS = {
    "rerank_text + bm25_facts": ["rerank_text", "bm25_facts"],
    "rerank_text + bm25_facts + bm25_text": ["rerank_text", "bm25_facts", "bm25_text"],
    "rerank_facts + bm25_facts": ["rerank_facts", "bm25_facts"],
    "rerank_text + rerank_facts": ["rerank_text", "rerank_facts"],
    "rerank_text + rerank_facts + bm25_facts": ["rerank_text", "rerank_facts", "bm25_facts"],
}


def fact_block(title: str, label: dict) -> str:
    """One document's extraction as one text: title, about, entities, facts."""
    return "\n".join(
        [title, label["about"], "Entities: " + ", ".join(label["entities"]), *label["facts"]]
    )


def ranks(scores: dict[str, float]) -> dict[str, int]:
    """Rank of every document by score, 1 = best."""
    ordered = sorted(scores, key=lambda doc_id: -scores[doc_id])

    return {doc_id: position for position, doc_id in enumerate(ordered, 1)}


def fuse(*rankings: dict[str, int], weights: list[float] | None = None) -> dict[str, float]:
    """Reciprocal rank fusion, each ranking weighted (1 each by default)."""
    weights = weights or [1.0] * len(rankings)

    return {
        d: sum(w / (RRF_K + ranking[d]) for w, ranking in zip(weights, rankings, strict=True))
        for d in rankings[0]
    }


def rerank_scores(base: dict[str, dict]) -> dict[str, dict]:
    """Reranker scores of every candidate on its text and on its fact block; cached."""
    if SCORES.exists():
        return json.loads(SCORES.read_text())
    documents = json.loads(POOLS.read_text())
    pools, texts = documents["pools"], documents["documents"]
    labels = {v["doc_id"]: v for v in json.loads(LABELS.read_text()).values()}
    reranker = GteModernBertReranker(batch_size=4)
    out: dict[str, dict] = {}
    seconds = {"text": 0.0, "facts": 0.0}
    with RunProgress(get_engine(), "rerank pilot: score candidates", len(base)) as progress:
        for qid, item in base.items():
            ids = list(item["scores"]["gte_text"])
            question = normalize_text(pools[qid]["question"])
            started = time.perf_counter()
            on_text = reranker.score(question, [texts[d] for d in ids])
            seconds["text"] += time.perf_counter() - started
            started = time.perf_counter()
            blocks = [fact_block(texts[d].split("\n", 1)[0], labels[d]) for d in ids]
            on_facts = reranker.score(question, blocks)
            seconds["facts"] += time.perf_counter() - started
            out[qid] = {
                "rerank_text": dict(zip(ids, on_text, strict=True)),
                "rerank_facts": dict(zip(ids, on_facts, strict=True)),
            }
            progress.advance("questions", 1, note=qid)
    documents_scored = sum(len(v["rerank_text"]) for v in out.values())
    out["_timing"] = {
        "documents": documents_scored,
        "ms_per_doc_text": 1000 * seconds["text"] / documents_scored,
        "ms_per_doc_facts": 1000 * seconds["facts"] / documents_scored,
    }
    SCORES.write_text(json.dumps(out))

    return out


def two_stage(first: dict[str, int], rerank: dict[str, int], expected: str, k: int) -> int:
    """Rank after reranking only the first stage's top k; others keep their order below."""
    if first[expected] > k:
        return first[expected]
    kept = {d: rerank[d] for d, r in first.items() if r <= k}

    return ranks({d: -r for d, r in kept.items()})[expected]


def report(name: str, values: list[int]) -> None:
    """One table line: questions found in the top 1/5/10, recall and misses, median rank."""
    values = sorted(values)
    n = len(values)
    top5, top10 = sum(v <= 5 for v in values), sum(v <= 10 for v in values)
    print(
        f"{name:48} {sum(v <= 1 for v in values):>6} {top5:>6} {top10:>7} "
        f"{100 * top5 / n:>8.0f}% {100 * top10 / n:>8.0f}% {n - top10:>7} {values[n // 2]:>7}"
    )


def main() -> None:
    """Score, fuse and report where the expected document lands among its candidates."""
    base = json.loads(BASE_SCORES.read_text())
    reranked = rerank_scores(base)
    rows = []
    for qid, item in base.items():
        scores = {**item["scores"], **reranked[qid]}
        ranked = {name: ranks(s) for name, s in scores.items()}
        expected = item["expected"]
        row: dict[str, object] = {"question_id": qid}
        for name in ["bm25_text", "bm25_facts", "rerank_text", "rerank_facts"]:
            row[name] = ranked[name][expected]
        first = ranks(fuse(*[ranked[p] for p in FIRST_STAGE], weights=FIRST_WEIGHTS))
        row["first stage (bm25 text+facts, gte 0.25)"] = first[expected]
        for name, parts in FUSIONS.items():
            row[name] = ranks(fuse(*[ranked[p] for p in parts]))[expected]
        for weight in WEIGHTS:
            fused = fuse(ranked["rerank_text"], ranked["bm25_facts"], weights=[1.0, weight])
            row[f"rerank_text + {weight}*bm25_facts"] = ranks(fused)[expected]
        for k in (10, 20, 30):
            for which in ("rerank_text", "rerank_facts"):
                row[f"first stage top {k} -> {which}"] = two_stage(
                    first, ranked[which], expected, k
                )
        rows.append(row)

    OUT.write_text(json.dumps(rows, indent=1))
    timing = reranked["_timing"]
    print(
        f"\nreranker: {timing['ms_per_doc_text']:.0f} ms/doc on text, "
        f"{timing['ms_per_doc_facts']:.0f} ms/doc on facts ({timing['documents']} documents)"
    )
    print(f"\n{len(rows)} questions; rank of the expected document among its ~51 candidates")
    print(
        f"{'variant':48} {'top 1':>6} {'top 5':>6} {'top 10':>7} {'recall@5':>9} "
        f"{'recall@10':>9} {'missed':>7} {'median':>7}"
    )
    for name in [k for k in rows[0] if k != "question_id"]:
        report(name, [int(row[name]) for row in rows])  # type: ignore[call-overload]
    print(f"wrote {OUT}")


main()
