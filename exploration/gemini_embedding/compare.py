"""Do Gemini embeddings spread gte's near-duplicates further apart?

Re-scores the cached gte top 50 of 18 questions (notebook 008's cache,
`exploration/embedding_map/data/top50.json`) with gemini-embedding-2 and compares,
on the same documents: rank 1, rank 50 and the width of the band; where the right
document lands; own vs another question's top 50; and how far the two orders agree.
It doesn't search the corpus with Gemini, so it says nothing about Gemini's recall.

gemini-embedding-2 merges a list of inputs into one vector, so every text is its own
call. Vectors are cached in data/ (resumable). About $0.30.

Usage:
    uv run --env-file .env python exploration/gemini_embedding/compare.py
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from scipy.stats import spearmanr
from sqlalchemy import text
from tenacity import retry, stop_after_attempt, wait_exponential

from train_rag.database import get_engine
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress
from train_rag.tables import EMBEDDING_TABLES

MODEL = "gemini-embedding-2"
PRICE_PER_MILLION = 0.20
CHARS_PER_TOKEN = 3.8
MAX_CHARS = 30_000  # about 8k tokens, the model's input limit
TOP = 50
WORKERS = 8
TOP50 = Path(__file__).parent.parent / "embedding_map/data/top50.json"
DATA = Path(__file__).parent / "data"
CACHE = DATA / "gemini_vectors.json"
RESULT = DATA / "comparison.json"


@retry(stop=stop_after_attempt(6), wait=wait_exponential(min=2, max=60))
def embed_one(client: genai.Client, content: str, task_type: str) -> list[float]:
    """Embed one text with Gemini, retrying on rate limits and server errors.

    Args:
        client: The Gemini client.
        content: The text.
        task_type: "RETRIEVAL_QUERY" or "RETRIEVAL_DOCUMENT".

    Returns:
        The embedding.
    """
    response = client.models.embed_content(
        model=MODEL,
        contents=content[:MAX_CHARS],
        config=types.EmbedContentConfig(task_type=task_type),
    )

    return list(response.embeddings[0].values)


def load_texts(doc_ids: list[str]) -> list[tuple[str, int, str]]:
    """Read the chunk texts gte embedded for these documents.

    Args:
        doc_ids: The documents.

    Returns:
        (doc_id, position, text) per chunk.
    """
    chunks: list[tuple[str, int, str]] = []
    with get_engine().connect() as connection:
        for table in EMBEDDING_TABLES[768].values():
            chunks += [
                (row[0], row[1], row[2])
                for row in connection.execute(
                    text(
                        f"select doc_id, position, text from {table.__tablename__} "
                        "where model = 'gte-modernbert-base' and field = 'chunk' "
                        "and doc_id = any(:ids)"
                    ),
                    {"ids": doc_ids},
                ).all()
            ]

    return chunks


def embed_all(items: dict[str, tuple[str, str]]) -> dict[str, list[float]]:
    """Embed every item not yet cached, in parallel, saving as it goes.

    Args:
        items: key -> (text, task type).

    Returns:
        key -> vector for every item.
    """
    cached: dict[str, list[float]] = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [key for key in items if key not in cached]
    chars = sum(min(len(items[key][0]), MAX_CHARS) for key in todo)
    print(
        f"{len(todo)} texts to embed, ~{chars / CHARS_PER_TOKEN / 1e6:.2f}M tokens, "
        f"~${chars / CHARS_PER_TOKEN / 1e6 * PRICE_PER_MILLION:.2f}"
    )
    client = genai.Client()
    DATA.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    with RunProgress(get_engine(), f"embed with {MODEL} (~$0.30)", len(todo), chars) as progress:
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = {key: pool.submit(embed_one, client, *items[key]) for key in todo}
            for count, (key, future) in enumerate(futures.items(), 1):
                cached[key] = future.result()
                size = min(len(items[key][0]), MAX_CHARS)
                spent += size / CHARS_PER_TOKEN / 1e6 * PRICE_PER_MILLION
                progress.advance("embedding", 1, size, note=f"~${spent:.2f} spent")
                if count % 100 == 0:
                    CACHE.write_text(json.dumps(cached))
    CACHE.write_text(json.dumps(cached))

    return cached


def unit(vectors: list[list[float]]) -> np.ndarray:
    """Scale vectors to unit length.

    Args:
        vectors: The vectors.

    Returns:
        The unit vectors as an array.
    """
    matrix = np.array(vectors, dtype=np.float64)

    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def main() -> None:
    """Embed the cached top 50s with Gemini, compare with gte, save and print."""
    neighbourhoods: dict[str, Any] = json.loads(TOP50.read_text())
    with get_engine().connect() as connection:
        expected = dict(
            connection.execute(
                text("select id, expected_doc_ids from questions where id = any(:ids)"),
                {"ids": list(neighbourhoods)},
            ).all()
        )
        question_texts = dict(
            connection.execute(
                text("select id, text from questions where id = any(:ids)"),
                {"ids": list(neighbourhoods)},
            ).all()
        )
    doc_ids = sorted(
        {doc["doc_id"] for value in neighbourhoods.values() for doc in value["documents"]}
    )
    chunks = load_texts(doc_ids)
    items = {
        f"q:{question_id}": (normalize_text(question_texts[question_id]), "RETRIEVAL_QUERY")
        for question_id in neighbourhoods
    }
    items |= {
        f"d:{doc_id}:{position}": (chunk_text, "RETRIEVAL_DOCUMENT")
        for doc_id, position, chunk_text in chunks
    }
    vectors = embed_all(items)

    chunk_keys: dict[str, list[str]] = {}
    for doc_id, position, _ in chunks:
        chunk_keys.setdefault(doc_id, []).append(f"d:{doc_id}:{position}")

    def gemini_scores(question_id: str, documents: list[dict[str, Any]]) -> np.ndarray:
        """Score documents for a question by their best chunk, with Gemini vectors."""
        query = unit([vectors[f"q:{question_id}"]])[0]

        return np.array(
            [
                max(unit([vectors[key] for key in chunk_keys[doc["doc_id"]]]) @ query)
                for doc in documents
            ]
        )

    rows = []
    for question_id, value in sorted(neighbourhoods.items()):
        documents = value["documents"]
        gte = np.array([1 - doc["distance"] for doc in documents])
        gemini = gemini_scores(question_id, documents)
        right = [
            index for index, doc in enumerate(documents) if doc["doc_id"] in expected[question_id]
        ]
        top_gte, top_gemini = np.sort(gte[:TOP])[::-1], np.sort(gemini[:TOP])[::-1]
        row = {
            "question": question_id,
            "gte rank1": top_gte[0],
            "gte rank50": top_gte[-1],
            "gemini rank1": top_gemini[0],
            "gemini rank50": top_gemini[-1],
            "order agreement (Spearman)": spearmanr(gte[:TOP], gemini[:TOP]).statistic,
        }
        if right:
            best_gte = max(right, key=lambda index: gte[index])
            best_gemini = max(right, key=lambda index: gemini[index])
            row |= {
                "right gte rank": int((gte > gte[best_gte]).sum()) + 1,
                "right gemini rank": int((gemini > gemini[best_gemini]).sum()) + 1,
                "right gte gap": top_gte[0] - gte[best_gte],
                "right gemini gap": max(top_gemini[0], gemini[best_gemini]) - gemini[best_gemini],
            }
        rows.append(row)

    # Own vs another question's top 50, for both models
    own = {"gte": [], "gemini": []}
    other = {"gte": [], "gemini": []}
    for question_id in neighbourhoods:
        for other_id, value in neighbourhoods.items():
            documents = value["documents"][:TOP]
            target = own if other_id == question_id else other
            target["gemini"].append(gemini_scores(question_id, documents).mean())
            question_vector = unit([neighbourhoods[question_id]["question_vector"]])[0]
            target["gte"].append(
                float(np.mean(unit([doc["vector"] for doc in documents]) @ question_vector))
            )

    RESULT.write_text(json.dumps({"rows": rows, "own": own, "other": other}, default=float))
    print_summary(rows, own, other)


def print_summary(
    rows: list[dict[str, Any]], own: dict[str, list[float]], other: dict[str, list[float]]
) -> None:
    """Print the per-question table and the averages.

    Args:
        rows: Per-question comparison.
        own: Mean score of each question's own top 50, per model.
        other: Mean score of other questions' top 50s, per model.
    """
    print(
        "\nquestion  | gte r1  r50  width | gemini r1  r50  width | Spearman | "
        "right rank gte→gemini, gap gte→gemini"
    )
    for row in rows:
        right = ""
        if "right gte rank" in row:
            right = (
                f"{row['right gte rank']:>3} → {row['right gemini rank']:<3} "
                f"gap {row['right gte gap']:.3f} → {row['right gemini gap']:.3f}"
            )
        print(
            f"{row['question']} | {row['gte rank1']:.3f} {row['gte rank50']:.3f} "
            f"{row['gte rank1'] - row['gte rank50']:.3f} | {row['gemini rank1']:.3f} "
            f"{row['gemini rank50']:.3f} {row['gemini rank1'] - row['gemini rank50']:.3f} | "
            f"{row['order agreement (Spearman)']:.2f} | {right}"
        )
    for model in ("gte", "gemini"):
        widths = [row[f"{model} rank1"] - row[f"{model} rank50"] for row in rows]
        print(
            f"\n{model}: median rank1 {np.median([row[f'{model} rank1'] for row in rows]):.3f}, "
            f"rank50 {np.median([row[f'{model} rank50'] for row in rows]):.3f}, "
            f"width {np.median(widths):.3f}; "
            f"own top 50 mean {np.mean(own[model]):.3f}, other questions' top 50 mean "
            f"{np.mean(other[model]):.3f}, gap {np.mean(own[model]) - np.mean(other[model]):.3f}"
        )


if __name__ == "__main__":
    main()
