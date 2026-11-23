"""If search knew the right source (e.g. "it was a pull request"), would it find the document?

For each of the 46 semantic failures, runs today's hybrid search (gte + BM25, RRF) twice:
over the whole corpus, and over only the expected document's source (GitHub, Jira, Slack,
...). gte searches only that source's table; BM25 runs deep over everything and keeps that
source's hits. Reports how many expected documents land in the fused top 10 / 50.

This is the best case for a source filter: the person names the source correctly.
Free: no API calls. Writes data/source_filter_pilot.json.

    uv run python exploration/clarification/source_filter_pilot.py
"""

import asyncio
import json
from pathlib import Path

from sqlalchemy import text

from train_rag.database import get_engine
from train_rag.pipelines import retrieval
from train_rag.pipelines.retrieval import RetrievalPipeline
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress

HERE = Path(__file__).parent
POOLS = HERE.parent / "retrieval_failures" / "data" / "pools_run15.json"
OUT = HERE / "data" / "source_filter_pilot.json"
DEPTH = 50
BM25_DEEP = 5_000  # BM25 is over the whole corpus; read deep, then keep one source
RRF_K = 60
CUTS = [5, 10, 50]


def fuse(gte: list[str], bm25: list[str]) -> list[str]:
    """Reciprocal rank fusion of two rankings, best first, cut to `DEPTH`."""
    scores: dict[str, float] = {}
    for ranking in (gte, bm25):
        for position, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (RRF_K + position)

    return sorted(scores, key=lambda doc_id: -scores[doc_id])[:DEPTH]


def rank_of(doc_ids: list[str], expected: str) -> int | None:
    """1-based position of the expected document, or None when it isn't in the list."""
    return doc_ids.index(expected) + 1 if expected in doc_ids else None


async def main() -> None:
    """Search every failure over the whole corpus and over its right source only."""
    pools = json.loads(POOLS.read_text())["pools"]
    with get_engine().connect() as connection:
        source_of = dict(
            connection.execute(
                text("select doc_id, source_type from documents where doc_id = any(:ids)"),
                {"ids": [pool["expected"] for pool in pools.values()]},
            ).all()
        )
    pipeline = RetrievalPipeline.load(rerank=False)
    all_sources = list(pipeline.sources)
    rows = []
    with RunProgress(get_engine(), "source filter pilot", len(pools)) as progress:
        for qid, pool in pools.items():
            clean = normalize_text(pool["question"])
            expected = pool["expected"]
            source = source_of[expected]

            retrieval.CANDIDATES = DEPTH
            pipeline.sources = all_sources
            everywhere = fuse(
                [h.doc_id for h in await pipeline.semantic_search(clean)],
                [h.doc_id for h in await pipeline.keyword_search(clean)],
            )

            pipeline.sources = [s for s in all_sources if str(s) == str(source)]
            gte = [h.doc_id for h in await pipeline.semantic_search(clean)]
            retrieval.CANDIDATES = BM25_DEEP
            bm25 = [
                h.doc_id
                for h in await pipeline.keyword_search(clean)
                if str(h.source_type) == str(source)
            ][:DEPTH]
            filtered = fuse(gte, bm25)

            rows.append(
                {
                    "question_id": qid,
                    "source": str(source),
                    "everywhere": rank_of(everywhere, expected),
                    "right_source": rank_of(filtered, expected),
                }
            )
            progress.advance("questions", 1, note=f"{qid} {source}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))

    print(f"{len(rows)} failures; expected document within the fused top N")
    print(f"{'N':>4} {'everywhere':>11} {'right source only':>18}")
    for cut in CUTS:
        a = sum(1 for r in rows if r["everywhere"] is not None and r["everywhere"] <= cut)
        b = sum(1 for r in rows if r["right_source"] is not None and r["right_source"] <= cut)
        print(f"{cut:>4} {a:>11} {b:>18}")
    by_source: dict[str, list[int]] = {}
    for r in rows:
        hit = r["right_source"] is not None and r["right_source"] <= 10
        by_source.setdefault(r["source"], []).append(int(hit))
    print("\ntop 10 with the right source, by source:")
    for source, hits in sorted(by_source.items(), key=lambda item: -len(item[1])):
        print(f"  {source:13} {sum(hits)} of {len(hits)}")
    print(f"wrote {OUT}")


asyncio.run(main())
