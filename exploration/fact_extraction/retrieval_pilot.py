r"""Do extracted facts help the search find the expected document among its near-duplicates?

For each of the 46 search failures of post 004, the pool is gte's top 50 plus the expected
document and the pipeline's picks (pools.py). Every pool document is scored for the question
in several ways, and the expected document's rank in its pool is recorded:

- gte_text: gte-modernbert on the full text in 2,000-character chunks, best chunk counts
  (the full-text baseline at the same pool);
- bm25_text: BM25 on the full text (statistics from the pool, so only roughly the corpus's);
- gte_facts: gte on one block per document: title, about, entities and facts;
- gte_each_fact: gte on the about line and every fact separately, best one counts;
- bm25_facts: BM25 on the fact block;
- rrf_text_facts: gte_text and gte_facts fused by reciprocal rank (facts added to the index,
  not replacing the text);
- rrf_all: all four of gte_text, bm25_text, gte_facts and bm25_facts fused;
- rrf_each_bm25facts: gte_each_fact and bm25_facts fused (an index of facts only);
- rrf_each_bm25facts_text: the same plus bm25_text.

Facts are DeepSeek's labels (data/labels.json, prompt 2a3743f7bb9a). Ranks are within the
pool, so they are optimistic for the whole corpus, as in the earlier pool tests.

Usage (from the main checkout, which has the data and the GPU code):
    uv run --project ~/projects/rag-implementation/train-rag python \
        exploration/fact_extraction/retrieval_pilot.py
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import Stemmer

from train_rag.database import get_engine
from train_rag.embedder import Embedder, GteModernBertEmbedder, MiniLmEmbedder
from train_rag.index_bm25 import tokenize
from train_rag.processing.chunking import RecursiveChunker
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress

HERE = Path(__file__).parent
POOLS = Path.home() / "projects/rag-implementation/train-rag/exploration/retrieval_failures/data"
POOLS = POOLS / "pools_run15.json"
LABELS = HERE / "data" / "labels.json"
OUT = HERE / "data" / "retrieval_pilot.json"
SCORES = HERE / "data" / "retrieval_pilot_scores.json"  # embedding is slow; fusions are not
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
CHUNK, CONTEXT = 2_000, 200
RRF_K = 60
STEMMER = Stemmer.Stemmer("english")
BASE = [
    "gte_text",
    "bm25_text",
    "gte_facts",
    "gte_each_fact",
    "bm25_facts",
    "minilm_text",
    "minilm_facts",
    "minilm_each_fact",
]
MINILM_CHUNK, MINILM_CONTEXT = 1_000, 100  # MiniLM reads about 1,000 characters at most
# Named fusions: the rankings fused, with equal weights.
FUSIONS = {
    "rrf_current (gte_text + bm25_text)": ["gte_text", "bm25_text"],
    "rrf_text_facts": ["gte_text", "gte_facts"],
    "rrf_all": ["gte_text", "bm25_text", "gte_facts", "bm25_facts"],
    "rrf_each_bm25facts": ["gte_each_fact", "bm25_facts"],
    "rrf_each_bm25facts_text": ["gte_each_fact", "bm25_facts", "bm25_text"],
    "rrf_minilm_each_bm25facts": ["minilm_each_fact", "bm25_facts"],
}
# Weighted fusions: these rankings at weight 1, plus a per-fact ranking at each of WEIGHTS.
WEIGHTED = {
    "bm25_facts + w*each": (["bm25_facts"], "gte_each_fact"),
    "bm25_text + bm25_facts + w*each": (["bm25_text", "bm25_facts"], "gte_each_fact"),
    "current + bm25_facts + w*each": (["gte_text", "bm25_text", "bm25_facts"], "gte_each_fact"),
    "bm25_facts + w*minilm_each": (["bm25_facts"], "minilm_each_fact"),
    "current + bm25_facts + w*minilm_each": (
        ["gte_text", "bm25_text", "bm25_facts"],
        "minilm_each_fact",
    ),
}


def fact_block(title: str, label: dict) -> str:
    """One document's extraction as one text: title, about, entities, facts."""
    return "\n".join(
        [title, label["about"], "Entities: " + ", ".join(label["entities"]), *label["facts"]]
    )


def bm25(query: str, texts: dict[str, str]) -> dict[str, float]:
    """Okapi BM25 (k1 1.5, b 0.75) of a query against a small collection."""
    docs = dict(zip(texts, tokenize(list(texts.values()), STEMMER), strict=True))
    average = sum(len(words) for words in docs.values()) / len(docs)
    frequency = Counter(word for words in docs.values() for word in set(words))
    terms = tokenize([query], STEMMER)[0]
    scores = {}
    for doc_id, words in docs.items():
        counts = Counter(words)
        score = 0.0
        for term in terms:
            if term not in counts:
                continue
            idf = math.log(1 + (len(docs) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            tf = counts[term]
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(words) / average))
        scores[doc_id] = score

    return scores


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


def embed_grouped(embedder: Embedder, groups: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Embed every text of every group in one pass; vectors come back per group."""
    flat = [text for texts in groups.values() for text in texts]
    vectors = np.array(embedder.embed(flat))
    out, start = {}, 0
    for key, texts in groups.items():
        out[key] = vectors[start : start + len(texts)]
        start += len(texts)

    return out


def pool_inputs() -> tuple[dict, dict, dict[str, str], dict[str, str], dict[str, list[str]]]:
    """Pools, document texts, titles, fact blocks and single facts of every pool document."""
    data = json.loads(POOLS.read_text())
    pools, documents = data["pools"], data["documents"]
    labels = {v["doc_id"]: v for v in json.loads(LABELS.read_text()).values()}
    members = sorted({d for pool in pools.values() for d in pool["members"]})
    titles = {d: documents[d].split("\n", 1)[0] for d in members}
    blocks = {d: fact_block(titles[d], labels[d]) for d in members if d in labels}
    each = {d: [labels[d]["about"], *labels[d]["facts"]] for d in blocks}

    return pools, documents, titles, blocks, each


def add_minilm(computed: dict[str, dict]) -> None:
    """Add MiniLM's scores (text in 1,000-character chunks, fact block, single facts)."""
    pools, documents, titles, blocks, each = pool_inputs()
    chunker = RecursiveChunker(size=MINILM_CHUNK, context=MINILM_CONTEXT)
    chunks = {
        d: chunker.chunks(titles[d], documents[d].split("\n", 1)[-1]) or [documents[d]]
        for d in blocks
    }
    embedder: Embedder = MiniLmEmbedder()
    questions = [normalize_text(pools[qid]["question"]) for qid in computed]
    queries = dict(zip(computed, np.array(embedder.embed(questions)), strict=True))
    text_vectors = embed_grouped(embedder, chunks)
    block_vectors = embed_grouped(embedder, {d: [b] for d, b in blocks.items()})
    fact_vectors = embed_grouped(embedder, each)
    print(f"MiniLM: {sum(map(len, chunks.values()))} text chunks embedded", flush=True)
    for qid, item in computed.items():
        ids, query = list(item["scores"]["gte_text"]), queries[qid]
        item["scores"]["minilm_text"] = {d: float(np.max(text_vectors[d] @ query)) for d in ids}
        item["scores"]["minilm_facts"] = {d: float(block_vectors[d][0] @ query) for d in ids}
        item["scores"]["minilm_each_fact"] = {
            d: float(np.max(fact_vectors[d] @ query)) for d in ids
        }


def compute_scores() -> dict[str, dict]:
    """Every pool document's score for its question, every base way; cached in SCORES."""
    if SCORES.exists():
        computed = json.loads(SCORES.read_text())
        if "minilm_each_fact" not in next(iter(computed.values()))["scores"]:
            add_minilm(computed)
            SCORES.write_text(json.dumps(computed))
        return computed
    data = json.loads(POOLS.read_text())
    pools, documents = data["pools"], data["documents"]
    labels = {v["doc_id"]: v for v in json.loads(LABELS.read_text()).values()}
    members = sorted({d for pool in pools.values() for d in pool["members"]})
    missing = [d for d in members if d not in labels]
    print(f"{len(pools)} questions, {len(members)} pool documents, {len(missing)} without label")

    titles = {d: documents[d].split("\n", 1)[0] for d in members}
    blocks = {d: fact_block(titles[d], labels[d]) for d in members if d in labels}
    each = {d: [labels[d]["about"], *labels[d]["facts"]] for d in blocks}
    chunker = RecursiveChunker(size=CHUNK, context=CONTEXT)
    chunks = {
        d: chunker.chunks(titles[d], documents[d].split("\n", 1)[-1]) or [documents[d]]
        for d in members
    }

    embedder = GteModernBertEmbedder(batch_size=16)
    progress = RunProgress(get_engine(), "fact retrieval pilot: embed pools", 4)
    with progress:
        queries = dict(
            zip(
                pools,
                np.array(embedder.embed([normalize_text(p["question"]) for p in pools.values()])),
                strict=True,
            )
        )
        progress.advance("questions", 1)
        text_vectors = embed_grouped(embedder, chunks)
        progress.advance("text chunks", 1, note=f"{sum(map(len, chunks.values()))} chunks")
        block_vectors = embed_grouped(embedder, {d: [b] for d, b in blocks.items()})
        progress.advance("fact blocks", 1)
        fact_vectors = embed_grouped(embedder, each)
        progress.advance("single facts", 1)

    out = {}
    for qid, pool in pools.items():
        ids = [d for d in pool["members"] if d in blocks]
        if pool["expected"] not in blocks:
            continue
        query, question = queries[qid], normalize_text(pool["question"])
        out[qid] = {
            "expected": pool["expected"],
            "scores": {
                "gte_text": {d: float(np.max(text_vectors[d] @ query)) for d in ids},
                "bm25_text": bm25(question, {d: documents[d] for d in ids}),
                "gte_facts": {d: float(block_vectors[d][0] @ query) for d in ids},
                "gte_each_fact": {d: float(np.max(fact_vectors[d] @ query)) for d in ids},
                "bm25_facts": bm25(question, {d: blocks[d] for d in ids}),
            },
        }
    add_minilm(out)
    SCORES.write_text(json.dumps(out))

    return out


def report(name: str, values: list[int]) -> None:
    """One table line: how often the expected document is at the top, and its median rank."""
    values = sorted(values)
    print(
        f"{name:42} {sum(v <= 1 for v in values):>6} {sum(v <= 5 for v in values):>6} "
        f"{sum(v <= 10 for v in values):>7} {values[len(values) // 2]:>7} "
        f"{sum(values) / len(values):>6.1f}"
    )


def main() -> None:
    """Score every pool every way and report where the expected document lands."""
    computed = compute_scores()
    rows = []
    for qid, item in computed.items():
        ranked = {name: ranks(s) for name, s in item["scores"].items()}
        row: dict[str, object] = {"question_id": qid}
        for name in BASE:
            row[name] = ranked[name][item["expected"]]
        for name, parts in FUSIONS.items():
            row[name] = ranks(fuse(*[ranked[p] for p in parts]))[item["expected"]]
        for name, (parts, vector) in WEIGHTED.items():
            for weight in WEIGHTS:
                fused = fuse(
                    *[ranked[p] for p in parts],
                    ranked[vector],
                    weights=[1.0] * len(parts) + [weight],
                )
                row[f"{name}, w={weight}"] = ranks(fused)[item["expected"]]
        rows.append(row)

    OUT.write_text(json.dumps(rows, indent=1))
    print(f"\n{len(rows)} questions; rank of the expected document in its pool (~51 documents)")
    print(f"{'variant':42} {'top 1':>6} {'top 5':>6} {'top 10':>7} {'median':>7} {'mean':>6}")
    for name in [k for k in rows[0] if k != "question_id"]:
        if name.endswith("w=0.0"):
            print()
        report(name, [int(row[name]) for row in rows])  # type: ignore[call-overload]
    print(f"wrote {OUT}")


main()
