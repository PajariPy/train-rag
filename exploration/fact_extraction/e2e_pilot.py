"""End-to-end: deep hybrid search, then the reranker, on the 46 semantic failures.

For each failure question:

1. Today's hybrid search (gte chunks + BM25, fused with RRF) with `DEPTH` documents from
   each ranking, keeping the fused top `DEPTH`.
2. Load the full text (title + text) of every fused document from Postgres.
3. Rerank all of them twice: on the full text (up to the reranker's 8,192 tokens) and on
   the DeepSeek fact block, or the first `NO_FACTS_CHARS` characters of the text where a
   document has no facts.

Reports recall@5, recall@10 and missed (not in the top 10) for: the fusion alone, today's
reranker depth (fused top 50 reranked on full text), and the 300-deep variants. Times
each stage per question.

Scores are cached per question in data/e2e_pilot_scores.json, so a rerun resumes.
Writes data/e2e_pilot.json. Free: no API calls.

    uv run python exploration/fact_extraction/e2e_pilot.py
"""

import asyncio
import json
import statistics
import time
from pathlib import Path

from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.pipelines import retrieval
from train_rag.pipelines.retrieval import RetrievalPipeline
from train_rag.processing.normalize import normalize_documents, normalize_text
from train_rag.progress.recorder import RunProgress
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.reranker import GteModernBertReranker

HERE = Path(__file__).parent
POOLS = HERE.parent / "retrieval_failures" / "data" / "pools_run15.json"
LABELS = HERE / "data" / "labels.json"
SCORES = HERE / "data" / "e2e_pilot_scores.json"
OUT = HERE / "data" / "e2e_pilot.json"
DEPTH = 300
TODAY_DEPTH = 50
NO_FACTS_CHARS = 2_000
RRF_K = 60


def fact_block(title: str, label: dict) -> str:
    """One document's extraction as one text: title, about, entities, facts."""
    return "\n".join(
        [title, label["about"], "Entities: " + ", ".join(label["entities"]), *label["facts"]]
    )


def fuse(gte: list[str], bm25: list[str]) -> list[str]:
    """Reciprocal rank fusion of two rankings, best first, cut to `DEPTH`."""
    scores: dict[str, float] = {}
    for ranking in (gte, bm25):
        for position, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (RRF_K + position)

    return sorted(scores, key=lambda doc_id: -scores[doc_id])[:DEPTH]


def load_texts(doc_ids: list[str]) -> dict[str, str]:
    """Full normalized title + text of documents, keyed by dataset id."""
    with Session(get_engine()) as session:
        internal = ids_of_doc_ids(session, EnterpriseRagBenchLoader.name, doc_ids)
        clean = normalize_documents(load_documents(session, sorted(internal)))

    return {doc.doc_id: f"{doc.title}\n\n{doc.text}" for doc in clean}


def rank_by(scores: dict[str, float], doc_ids: list[str], expected: str) -> int | None:
    """1-based rank of the expected document among `doc_ids` by score, or None if absent."""
    if expected not in doc_ids:
        return None
    ordered = sorted(doc_ids, key=lambda doc_id: -scores[doc_id])

    return ordered.index(expected) + 1


def rrf_rank(a: dict[str, float], b: dict[str, float], ids: list[str], expected: str) -> int | None:
    """Rank of the expected document after fusing two score sets by RRF."""
    if expected not in ids:
        return None
    rank_a = {d: r for r, d in enumerate(sorted(ids, key=lambda d: -a[d]), 1)}
    rank_b = {d: r for r, d in enumerate(sorted(ids, key=lambda d: -b[d]), 1)}
    fused = {d: 1 / (RRF_K + rank_a[d]) + 1 / (RRF_K + rank_b[d]) for d in ids}

    return rank_by(fused, ids, expected)


async def score_all(pools: dict[str, dict]) -> dict[str, dict]:
    """Search, load and rerank every failure; cached per question."""
    cache: dict[str, dict] = json.loads(SCORES.read_text()) if SCORES.exists() else {}
    todo = [qid for qid in pools if qid not in cache]
    if not todo:
        return cache
    labels = {v["doc_id"]: v for v in json.loads(LABELS.read_text()).values()}
    retrieval.CANDIDATES = DEPTH  # read at call time by both searches
    pipeline = RetrievalPipeline.load(rerank=False)
    reranker = GteModernBertReranker(batch_size=4)
    with RunProgress(get_engine(), "e2e pilot: search 300 deep + rerank", len(todo)) as progress:
        for qid in todo:
            clean = normalize_text(pools[qid]["question"])
            started = time.perf_counter()
            semantic, keyword = await asyncio.gather(
                pipeline.semantic_search(clean), pipeline.keyword_search(clean)
            )
            fused = fuse([h.doc_id for h in semantic], [h.doc_id for h in keyword])
            search_s = time.perf_counter() - started

            started = time.perf_counter()
            texts = load_texts(fused)
            load_s = time.perf_counter() - started
            fused = [d for d in fused if d in texts]

            started = time.perf_counter()
            on_text = reranker.score(clean, [texts[d] for d in fused])
            text_s = time.perf_counter() - started

            blocks = [
                fact_block(texts[d].split("\n", 1)[0], labels[d])
                if d in labels
                else texts[d][:NO_FACTS_CHARS]
                for d in fused
            ]
            started = time.perf_counter()
            on_facts = reranker.score(clean, blocks)
            facts_s = time.perf_counter() - started

            cache[qid] = {
                "fused": fused,
                "with_facts": sum(d in labels for d in fused),
                "rerank_text": dict(zip(fused, on_text, strict=True)),
                "rerank_facts": dict(zip(fused, on_facts, strict=True)),
                "seconds": {
                    "search": search_s,
                    "load": load_s,
                    "rerank_text": text_s,
                    "rerank_facts": facts_s,
                },
            }
            SCORES.write_text(json.dumps(cache))
            progress.advance(
                "questions", 1, note=f"{qid} text {text_s:.1f}s facts {facts_s:.1f}s"
            )
    await pipeline.close()

    return cache


def report(name: str, values: list[int | None]) -> dict[str, float]:
    """Print one table line and return its numbers."""
    n = len(values)
    top5 = sum(v is not None and v <= 5 for v in values)
    top10 = sum(v is not None and v <= 10 for v in values)
    in_list = sum(v is not None for v in values)
    print(
        f"{name:44} {in_list:>7} {top5:>6} {top10:>7} {100 * top5 / n:>8.0f}% "
        f"{100 * top10 / n:>9.0f}% {n - top10:>7}"
    )

    return {"in_list": in_list, "top5": top5, "top10": top10, "missed": n - top10}


def main() -> None:
    """Score everything, then report recall and timing."""
    pools = json.loads(POOLS.read_text())["pools"]
    cache = asyncio.run(score_all(pools))
    rows = []
    for qid, pool in pools.items():
        item, expected = cache[qid], pool["expected"]
        fused, top = item["fused"], item["fused"][:TODAY_DEPTH]
        position = fused.index(expected) + 1 if expected in fused else None
        text, facts = item["rerank_text"], item["rerank_facts"]
        rows.append(
            {
                "question_id": qid,
                "fused": position,
                "rerank text @50 (today)": rank_by(text, top, expected),
                "rerank facts @50": rank_by(facts, top, expected),
                "rerank text @300": rank_by(text, fused, expected),
                "rerank facts @300": rank_by(facts, fused, expected),
                "rerank text + facts (RRF) @300": rrf_rank(text, facts, fused, expected),
            }
        )
    OUT.write_text(json.dumps(rows, indent=1))

    print(f"\n{len(rows)} failures; expected document's place after each variant")
    print(
        f"{'variant':44} {'in list':>7} {'top 5':>6} {'top 10':>7} {'recall@5':>9} "
        f"{'recall@10':>10} {'missed':>7}"
    )
    for name in [k for k in rows[0] if k != "question_id"]:
        report(name, [row[name] for row in rows])

    seconds = [cache[qid]["seconds"] for qid in pools]
    print("\nseconds per question (median / mean):")
    for stage in ("search", "load", "rerank_text", "rerank_facts"):
        values = [s[stage] for s in seconds]
        print(f"  {stage:14} {statistics.median(values):6.2f} / {statistics.mean(values):6.2f}")
    with_facts = [cache[qid]["with_facts"] for qid in pools]
    total = sum(len(cache[qid]["fused"]) for qid in pools)
    print(f"\nfused documents with facts: {sum(with_facts)} of {total}")
    print(f"wrote {OUT}")


main()
