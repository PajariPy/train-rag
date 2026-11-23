"""Field search over a deeper pool: does it find the right threads today's search misses?

The tool an agent would get: search all labelled documents by their mined fields only
(gte similarity of subject, problem and names profiles, as in field_match.py). For each of
the 200 Slack dev questions, rank the whole labelled pool (~24,400 documents, all 200 right
threads included) and check where the right thread lands. Reported separately for the
questions whose right thread is already among the reranker's candidates and for those it
never reaches (the cases a gate before the reranker has to fix).

Caveat: the pool is small, so recall here is higher than in a corpus-wide search. The fair
comparison is "raw gte": the same fine-tuned gte on the question and the stored chunk
vectors of the same pool (best chunk per document), i.e. today's dense search on this pool.

Free and local (reuses the cached vectors). Usage:
    uv run python exploration/slack_extraction/field_pool.py
"""

import json
from typing import Any

import numpy as np
from field_match import BASELINE, HERE, embed_profiles, load_labels
from sqlalchemy import text

from train_rag.database import get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.models import SourceType
from train_rag.tables.embedding import FINETUNED_TABLES

RESULT = HERE / "data" / "field_pool_result.json"
DEPTHS = (10, 50, 100, 300, 1_000)
# Profile weights (subject, problem, names); "subject" alone is also reported.
MIXES = {"subject": (1.0, 0.0, 0.0), "subject+problem": (1.0, 0.5, 0.0), "all": (1.0, 0.5, 0.5)}


def raw_gte_scores(pool: list[str], questions: list[str], texts: dict[str, str]) -> np.ndarray:
    """Today's dense score on the pool: best stored chunk cosine per document.

    Args:
        pool: The pool doc ids.
        questions: The question doc ids, in output order.
        texts: The question text per doc id.

    Returns:
        A (questions, pool) matrix of cosine similarities.
    """
    position = {item: index for index, item in enumerate(pool)}
    query = FineTunedGteEmbedder(batch_size=64).embed([texts[q] for q in questions])
    q_matrix = np.asarray(query, dtype=np.float32)
    q_matrix /= np.linalg.norm(q_matrix, axis=1, keepdims=True)
    best = np.full((len(questions), len(pool)), -1.0, dtype=np.float32)

    with get_engine().connect() as connection:
        for source in SourceType:
            table = FINETUNED_TABLES[source].__tablename__
            rows = connection.execute(
                text(f"SELECT doc_id, embedding::text FROM {table} WHERE doc_id = ANY(:ids)"),
                {"ids": pool},
            ).all()

            if not rows:
                continue

            columns = np.array([position[row[0]] for row in rows])
            chunks = np.array([json.loads(row[1]) for row in rows], dtype=np.float32)
            chunks /= np.linalg.norm(chunks, axis=1, keepdims=True) + 1e-9
            similarity = q_matrix @ chunks.T  # (questions, chunks)

            for index, column in enumerate(columns):
                np.maximum(best[:, column], similarity[:, index], out=best[:, column])

    return best


def main() -> None:
    """Rank the labelled pool by fields for every question and report recall by depth."""
    baseline: dict[str, dict[str, Any]] = json.loads(BASELINE.read_text())
    labels = load_labels()
    vectors = embed_profiles(labels)

    pool = sorted(item for item in labels if not item.startswith("question:"))
    matrix = np.stack([vectors[item] for item in pool])  # (pool, 3, dim)
    position = {item: index for index, item in enumerate(pool)}

    groups = {"reached": [], "missed": []}
    for doc_id, entry in baseline.items():
        if f"question:{doc_id}" in labels and doc_id in position:
            groups["reached" if doc_id in entry["scores"] else "missed"].append(doc_id)

    report: dict[str, Any] = {"pool": len(pool), **{k: len(v) for k, v in groups.items()}}
    questions = groups["reached"] + groups["missed"]
    rows = [json.loads(line) for line in (HERE.parents[1] / "exploration" / "training_questions"
                                          / "data" / "questions.jsonl").open()]
    texts = {row["doc_id"]: row["question"] for row in rows if row["split"] == "dev_slack"}
    raw = raw_gte_scores(pool, questions, texts)
    raw_row = {q: index for index, q in enumerate(questions)}

    def z(values: np.ndarray) -> np.ndarray:
        return (values - values.mean()) / (values.std() + 1e-9)

    for mix, weights in {"raw gte": None, **MIXES, "raw gte + all fields": MIXES["all"]}.items():
        ranks: dict[str, list[int]] = {"reached": [], "missed": []}

        for group, questions in groups.items():
            for doc_id in questions:
                q_vec = vectors[f"question:{doc_id}"]
                dense = raw[raw_row[doc_id]]

                if weights is None:
                    scores = dense
                else:
                    scores = sum(
                        weight * (matrix[:, column] @ q_vec[column])
                        for column, weight in enumerate(weights)
                        if weight
                    )

                    if mix.startswith("raw gte"):
                        scores = z(dense) + z(scores)

                target = scores[position[doc_id]]
                ranks[group].append(int((scores > target).sum()) + 1)

        report[mix] = {
            group: {
                f"top{depth}": round(100 * np.mean(np.array(values) <= depth), 1)
                for depth in DEPTHS
            }
            | {"median rank": int(np.median(values))}
            for group, values in ranks.items()
        }

    RESULT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
