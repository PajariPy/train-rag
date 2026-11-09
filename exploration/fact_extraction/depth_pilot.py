"""How deep must the reranker read to see the right document, for the 46 semantic failures?

Runs today's hybrid search over the whole corpus (gte-modernbert chunks + BM25, fused with
RRF), but takes `DEPTH` documents from each ranking instead of 50, and records where the
expected document lands in gte, in BM25 and in the fusion. If it sits at fused rank 120,
a reranker that reads 150 fused documents gets a chance at it.

Writes data/depth_pilot.json. Free: no API calls.

    uv run python exploration/fact_extraction/depth_pilot.py
"""

import asyncio
import json
from pathlib import Path

from train_rag.database import get_engine
from train_rag.pipelines import retrieval
from train_rag.pipelines.retrieval import RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress

HERE = Path(__file__).parent
POOLS = HERE.parent / "retrieval_failures" / "data" / "pools_run15.json"
OUT = HERE / "data" / "depth_pilot.json"
DEPTH = 1_000
RRF_K = 60
CUTS = [50, 100, 200, 300, 500, 1_000]


def rank_of(doc_ids: list[str], expected: str) -> int | None:
    """1-based position of the expected document, or None when it isn't in the list."""
    return doc_ids.index(expected) + 1 if expected in doc_ids else None


async def main() -> None:
    """Search every failure question deep and record the expected document's ranks."""
    retrieval.CANDIDATES = DEPTH  # read at call time by both searches
    pools = json.loads(POOLS.read_text())["pools"]
    pipeline = RetrievalPipeline.load(rerank=False)
    rows = []
    with RunProgress(get_engine(), "depth pilot: deep hybrid search", len(pools)) as progress:
        for qid, pool in pools.items():
            clean = normalize_text(pool["question"])
            semantic = await pipeline.semantic_search(clean)
            keyword = await pipeline.keyword_search(clean)
            gte = [hit.doc_id for hit in semantic]
            bm25 = [hit.doc_id for hit in keyword]
            scores: dict[str, float] = {}
            for ranking in (gte, bm25):
                for position, doc_id in enumerate(ranking, 1):
                    scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (RRF_K + position)
            fused = sorted(scores, key=lambda doc_id: -scores[doc_id])
            expected = pool["expected"]
            rows.append(
                {
                    "question_id": qid,
                    "expected": expected,
                    "gte": rank_of(gte, expected),
                    "bm25": rank_of(bm25, expected),
                    "fused": rank_of(fused, expected),
                    "fused_top": fused[:300],
                }
            )
            progress.advance("questions", 1, note=qid)
    OUT.write_text(json.dumps(rows, indent=1))

    print(f"{len(rows)} questions; expected document within the top N of each ranking")
    print(f"{'N':>6} {'gte':>5} {'bm25':>5} {'fused':>6}")
    for cut in CUTS:
        line = [
            sum(1 for r in rows if r[k] is not None and r[k] <= cut)
            for k in ("gte", "bm25", "fused")
        ]
        print(f"{cut:>6} {line[0]:>5} {line[1]:>5} {line[2]:>6}")
    print(f"wrote {OUT}")


asyncio.run(main())
