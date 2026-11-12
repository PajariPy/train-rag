"""Compare gte-modernbert-base with Qwen3-Embedding-0.6B on a sample, without a full re-index.

Design (estimating each model's rank of the right document over the whole corpus):

1. Questions: 100 answerable questions, stratified by type (seed 7).
2. Pool: every document in those questions' replayed gte top 50 and BM25 top 50
   (`exploration/retriever_attribution/data/attribution.json`), plus the right documents.
3. Distractors: 5,000 random corpus documents outside the pool.
4. Both models embed the questions and the same stored chunk texts gte indexed; a document
   scores by its best chunk, like the pipeline.
5. Estimated corpus rank of a right document =
   (pool documents scoring higher) + 1 + (distractors scoring higher) × (outside docs / 5,000).

gte's real top 50 lies inside the pool, so its estimate can be checked against the replay.
Qwen vectors are cached in data/ (resumable).

Usage:
    uv run python exploration/embedder_comparison/compare.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import Connection, text

from train_rag.database import get_engine
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress
from train_rag.tables import EMBEDDING_TABLES

SEED = 7
QUESTIONS = 100
DISTRACTORS = 5_000
BATCH = 4
MAX_TOKENS = 8192
QWEN = "Qwen/Qwen3-Embedding-0.6B"
DATA = Path(__file__).parent / "data"
ATTRIBUTION = Path(__file__).parent.parent / "retriever_attribution/data/attribution.json"
QWEN_CACHE = DATA / "qwen_chunks.npz"
RESULT = DATA / "comparison.json"
SAVE_EVERY = 400


def pick_questions(connection: Connection, replayed: dict[str, Any]) -> list[dict[str, Any]]:
    """Choose answerable questions in proportion to their type.

    Args:
        connection: Open database connection.
        replayed: The replayed rankings per question id.

    Returns:
        The chosen questions with id, type, text and right documents.
    """
    rows = connection.execute(
        text("select id, question_type, text, expected_doc_ids from questions")
    ).all()
    by_type: dict[str, list[Any]] = defaultdict(list)
    for row in sorted(rows, key=lambda row: row.id):
        if row.id in replayed and row.expected_doc_ids:
            by_type[row.question_type].append(row)
    total = sum(len(group) for group in by_type.values())
    rng = random.Random(SEED)
    chosen = []
    for group in by_type.values():
        chosen += rng.sample(group, max(1, round(QUESTIONS * len(group) / total)))

    return [
        {
            "id": row.id,
            "type": row.question_type,
            "text": row.text,
            "expected": row.expected_doc_ids,
        }
        for row in chosen
    ]


def load_chunks(connection: Connection, doc_ids: list[str]) -> list[tuple[str, int, str, str]]:
    """Read the stored gte chunks of some documents.

    Args:
        connection: Open database connection.
        doc_ids: The documents.

    Returns:
        (doc_id, position, text, gte vector as text) for every chunk.
    """
    chunks = []
    for table in EMBEDDING_TABLES[768].values():
        chunks += connection.execute(
            text(
                f"select doc_id, position, text, embedding::text from {table.__tablename__} "
                "where model = 'gte-modernbert-base' and field = 'chunk' and doc_id = any(:ids)"
            ),
            {"ids": doc_ids},
        ).all()

    return [(row[0], row[1], row[2], row[3]) for row in chunks]


def embed_chunks(model: SentenceTransformer, texts: list[str], keys: list[str]) -> np.ndarray:
    """Embed chunk texts with Qwen, longest-alike batches, cached and resumable.

    Args:
        model: The loaded Qwen embedder.
        texts: Chunk texts.
        keys: A unique key per chunk ("doc_id:position").

    Returns:
        One unit vector per chunk, in input order.
    """
    cached: dict[str, np.ndarray] = {}
    if QWEN_CACHE.exists():
        stored = np.load(QWEN_CACHE, allow_pickle=True)
        cached = dict(zip(stored["keys"], stored["vectors"], strict=True))
    DATA.mkdir(parents=True, exist_ok=True)
    todo = sorted(
        (index for index, key in enumerate(keys) if key not in cached), key=lambda i: len(texts[i])
    )
    chars = sum(len(texts[index]) for index in todo)
    with RunProgress(
        get_engine(),
        "embed comparison pool with Qwen3-Embedding-0.6B (free)",
        len(todo),
        chars,
        unit="chunks",
    ) as progress:
        for start in range(0, len(todo), SAVE_EVERY):
            batch = todo[start : start + SAVE_EVERY]
            vectors = model.encode(
                [texts[index] for index in batch], batch_size=BATCH, normalize_embeddings=True
            )
            cached.update(
                {keys[index]: vector for index, vector in zip(batch, vectors, strict=True)}
            )
            np.savez(
                QWEN_CACHE,
                keys=np.array(list(cached), dtype=object),
                vectors=np.stack(list(cached.values())),
            )
            progress.advance(
                "embedding", len(batch), sum(len(texts[index]) for index in batch), note="local, $0"
            )

    return np.stack([cached[key] for key in keys])


def document_scores(
    chunk_scores: np.ndarray, chunk_docs: list[str], doc_ids: list[str]
) -> np.ndarray:
    """Score each document by its best chunk.

    Args:
        chunk_scores: (questions, chunks) cosine similarities.
        chunk_docs: The document of each chunk.
        doc_ids: The documents, in output order.

    Returns:
        (questions, documents) scores.
    """
    column = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    scores = np.full((chunk_scores.shape[0], len(doc_ids)), -np.inf)
    for chunk, doc_id in enumerate(chunk_docs):
        scores[:, column[doc_id]] = np.maximum(scores[:, column[doc_id]], chunk_scores[:, chunk])

    return scores


def estimate_ranks(
    scores: np.ndarray,
    doc_ids: list[str],
    pool: set[str],
    questions: list[dict[str, Any]],
    outside: int,
) -> list[dict[str, float]]:
    """Estimate each right document's rank over the whole corpus.

    Args:
        scores: (questions, documents) scores over pool + distractors.
        doc_ids: The documents of the score columns.
        pool: The pool's doc ids (the rest are distractors).
        questions: The questions, in score-row order.
        outside: Corpus documents outside the pool.

    Returns:
        Per question, the estimated rank of each right document.
    """
    in_pool = np.array([doc_id in pool for doc_id in doc_ids])
    column = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    scale = outside / int((~in_pool).sum())
    ranks = []
    for row, question in enumerate(questions):
        per_doc = {}
        for doc_id in question["expected"]:
            if doc_id not in column:
                continue
            value = scores[row, column[doc_id]]
            above_pool = int((scores[row, in_pool] > value).sum())
            above_outside = int((scores[row, ~in_pool] > value).sum())
            per_doc[doc_id] = above_pool + 1 + above_outside * scale
        ranks.append(per_doc)

    return ranks


def main() -> None:
    """Build the pool, embed it with both models, estimate ranks, and save and print the result."""
    replayed = json.loads(ATTRIBUTION.read_text())
    engine = get_engine()
    with engine.connect() as connection:
        questions = pick_questions(connection, replayed)
        pool = set()
        for question in questions:
            rankings = replayed[question["id"]]
            pool |= set(rankings["gte"]) | set(rankings["bm25"]) | set(question["expected"])
        corpus_size = connection.execute(text("select count(*) from documents")).scalar_one()
        distractors = [
            row[0]
            for row in connection.execute(
                text(
                    "select doc_id from documents where not doc_id = any(:pool) "
                    "order by md5(doc_id || :seed) limit :limit"
                ),
                {"pool": list(pool), "seed": str(SEED), "limit": DISTRACTORS},
            ).all()
        ]
        chunks = load_chunks(connection, sorted(pool) + distractors)
    print(
        f"{len(questions)} questions, {len(pool):,} pool docs, "
        f"{len(distractors):,} distractors, {len(chunks):,} chunks"
    )

    chunk_docs = [chunk[0] for chunk in chunks]
    doc_ids = sorted(set(chunk_docs))
    question_texts = [normalize_text(question["text"]) for question in questions]
    gte_chunks = np.stack([np.fromstring(chunk[3][1:-1], sep=",") for chunk in chunks])
    gte_model = SentenceTransformer(
        "Alibaba-NLP/gte-modernbert-base", model_kwargs={"torch_dtype": torch.float16}
    )
    gte_questions = gte_model.encode(question_texts, normalize_embeddings=True)
    del gte_model
    torch.cuda.empty_cache()

    qwen_model = SentenceTransformer(QWEN, model_kwargs={"torch_dtype": torch.float16})
    qwen_model.max_seq_length = MAX_TOKENS
    qwen_questions = qwen_model.encode(
        question_texts, prompt_name="query", normalize_embeddings=True
    )
    qwen_chunks = embed_chunks(
        qwen_model, [chunk[2] for chunk in chunks], [f"{chunk[0]}:{chunk[1]}" for chunk in chunks]
    )

    outside = corpus_size - len(pool)
    results: dict[str, Any] = {"questions": questions, "models": {}}
    for name, question_vectors, chunk_vectors in (
        ("gte", gte_questions, gte_chunks),
        ("qwen", qwen_questions, qwen_chunks),
    ):
        scores = document_scores(
            np.asarray(question_vectors) @ np.asarray(chunk_vectors).T, chunk_docs, doc_ids
        )
        results["models"][name] = estimate_ranks(scores, doc_ids, pool, questions, outside)
    DATA.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(results))
    summarise(results, replayed)


def summarise(results: dict[str, Any], replayed: dict[str, Any]) -> None:
    """Print recall at 10 and 50 per model, gte's check against the replay, and what Qwen adds.

    Args:
        results: Questions and each model's estimated ranks.
        replayed: The replayed gte and BM25 top 50 per question.
    """
    questions = results["questions"]
    print("\nright documents with estimated corpus rank ≤ k:")
    for name, ranks in results["models"].items():
        values = [rank for per_doc in ranks for rank in per_doc.values()]
        print(
            f"  {name:5s} @10 {np.mean([v <= 10 for v in values]):6.1%}  "
            f"@50 {np.mean([v <= 50 for v in values]):6.1%}  "
            f"median rank {np.median(values):,.0f}  (n={len(values)})"
        )

    agree = total = 0
    gained: dict[str, int] = defaultdict(int)
    by_type: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for question, gte_ranks, qwen_ranks in zip(
        questions, results["models"]["gte"], results["models"]["qwen"], strict=True
    ):
        rankings = replayed[question["id"]]
        for doc_id, gte_rank in gte_ranks.items():
            total += 1
            agree += (gte_rank <= 50) == (doc_id in rankings["gte"])
            qwen_top = qwen_ranks[doc_id] <= 50
            by_type[question["type"]]["gte"].append(gte_rank <= 50)
            by_type[question["type"]]["qwen"].append(qwen_top)
            if qwen_top and doc_id not in rankings["gte"]:
                gained["bm25 only" if doc_id in rankings["bm25"] else "neither"] += 1
    print(f"\ngte estimate agrees with the real gte top 50 for {agree}/{total} right documents")
    print(f"Qwen top 50 finds right docs gte missed: {dict(gained)}")
    print("\nper type, right docs in top 50 (gte / qwen):")
    for question_type, values in sorted(by_type.items()):
        print(
            f"  {question_type:25s} {sum(values['gte']):3d} / {sum(values['qwen']):3d}  "
            f"of {len(values['gte'])}"
        )


if __name__ == "__main__":
    main()
