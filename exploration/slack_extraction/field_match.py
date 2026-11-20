"""Field-match test: do DeepSeek's mined fields re-order the reranked candidates better?

For each of the 200 Slack dev questions, the candidates and their reranker scores come from
slack_dev_baseline.json (no new search). Each candidate is compared with the question on its
mined fields (data/labels.jsonl, prompt fc8ae8878988):

- names: share of the question's name values (any name field) found in the candidate's;
- names_emb: gte similarity of the joined name values (paraphrased names still match);
- topic: gte similarity of "topic; concepts" strings;
- problem: gte similarity of "bug; cause; fix; question" strings;
- area: Jaccard overlap of the area labels.

New score = z(reranker) + weighted feature sum. Weights are picked by grid search on half A
(data/question_halves.json) and reported on half B, so the gain isn't fitted to the test.

Free and local (GPU for gte). Usage:
    uv run python exploration/slack_extraction/field_match.py
"""

import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from schema import FIELDS

from train_rag.embedder import FineTunedGteEmbedder

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
BASELINE = ROOT / "exploration" / "query_expansion" / "data" / "slack_dev_baseline.json"
LABELS = HERE / "data" / "labels.jsonl"
HALVES = HERE / "data" / "question_halves.json"
CACHE = HERE / "data" / "field_match_vectors.npz"  # (subject, problem, names) per item
RESULT = HERE / "data" / "field_match_result.json"
PROMPT = "fc8ae8878988"
NAMES = [name for name, (form, _) in FIELDS.items() if form == "name" and name != "doc_date"]
FEATURES = ("names", "names_emb", "topic", "problem", "area")
GRID = (0.0, 0.05, 0.1, 0.2, 0.4)  # 0.25+ hurt in the first run


def normalise(value: str) -> str:
    """Lower-case a name and drop punctuation, for word-for-word matching."""
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def profiles(label: dict[str, Any]) -> tuple[str, str, str]:
    """The strings embedded per item: subject ("topic; concepts"), problem and names."""
    fields = label["fields"]
    subject = "; ".join(fields["topic"] + fields["concept"])
    problem = "; ".join(fields["bug"] + fields["cause"] + fields["fix"] + fields["question"])
    names = "; ".join(value for field in NAMES for value in fields[field])

    return subject, problem, names


def load_labels() -> dict[str, dict[str, Any]]:
    """The labels of the frozen prompt, by item id (questions as `question:<doc_id>`)."""
    labels: dict[str, dict[str, Any]] = {}

    with LABELS.open() as file:
        for line in file:
            row = json.loads(line)

            if row["key"].endswith(PROMPT):
                labels[row["id"]] = row

    return labels


def embed_profiles(labels: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """Unit gte vectors (subject, problem, names) per item id, cached in data/."""
    ids = sorted(labels)

    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=False)

        if list(cached["ids"]) == ids and cached["vectors"].shape[1] == 3:
            return dict(zip(ids, cached["vectors"], strict=True))

    texts = [text or "none" for item in ids for text in profiles(labels[item])]
    print(f"embedding {len(texts):,} profile strings with gte", flush=True)
    vectors = np.asarray(FineTunedGteEmbedder(batch_size=256).embed(texts), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    vectors = vectors.reshape(len(ids), 3, -1)

    # Empty profiles get a zero vector, so they match nothing.
    for row, item in enumerate(ids):
        for column, text in enumerate(profiles(labels[item])):
            if not text:
                vectors[row, column] = 0

    np.savez(CACHE, ids=np.array(ids), vectors=vectors)

    return dict(zip(ids, vectors, strict=True))


def name_set(label: dict[str, Any]) -> set[str]:
    """All name values of an item, normalised."""
    return {normalise(value) for field in NAMES for value in label["fields"][field]} - {""}


def features(
    question: dict[str, Any], candidate: dict[str, Any], q_vec: np.ndarray, c_vec: np.ndarray
) -> list[float]:
    """The field-match features (FEATURES order) of one candidate for one question."""
    q_names = name_set(question)
    names = len(q_names & name_set(candidate)) / len(q_names) if q_names else 0.0
    q_area, c_area = set(question["labels"]["area"]), set(candidate["labels"]["area"])
    area = len(q_area & c_area) / len(q_area | c_area) if q_area | c_area else 0.0

    similarity = [float(q_vec[column] @ c_vec[column]) for column in range(3)]

    return [names, similarity[2], similarity[0], similarity[1], area]


def rank_of(target: str, ids: list[str], scores: np.ndarray) -> int | None:
    """1-based rank of the right thread, or None if it isn't a candidate."""
    if target not in ids:
        return None

    order = np.argsort(-scores, kind="stable")

    return int(np.where(order == ids.index(target))[0][0]) + 1


def hits(ranks: list[int | None]) -> dict[str, float]:
    """Top 1 / 5 / 10 in % of the questions."""
    return {
        f"top{k}": round(100 * sum(1 for r in ranks if r is not None and r <= k) / len(ranks), 1)
        for k in (1, 5, 10)
    }


def main() -> None:
    """Build the features, pick weights on half A, report on half B and all 200."""
    baseline: dict[str, dict[str, Any]] = json.loads(BASELINE.read_text())
    halves = json.loads(HALVES.read_text())
    labels = load_labels()
    vectors = embed_profiles(labels)

    # Per question: candidate ids, z-scored reranker scores, feature matrix.
    data: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    missing = 0

    for doc_id, entry in baseline.items():
        question = labels.get(f"question:{doc_id}")

        if question is None:
            continue

        ids = [item for item in entry["scores"] if item in labels]
        missing += len(entry["scores"]) - len(ids)
        rerank = np.array([entry["scores"][item] for item in ids], dtype=np.float64)
        rerank = (rerank - rerank.mean()) / (rerank.std() + 1e-9)
        q_vec = vectors[f"question:{doc_id}"]
        matrix = np.array(
            [features(question, labels[item], q_vec, vectors[item]) for item in ids]
        )
        matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-9)
        data[doc_id] = (ids, rerank, matrix)

    print(f"{len(data)} questions, {missing} candidates without labels")

    def evaluate(questions: list[str], weights: np.ndarray) -> dict[str, float]:
        ranks = []

        for doc_id in questions:
            ids, rerank, matrix = data[doc_id]
            ranks.append(rank_of(doc_id, ids, rerank + matrix @ weights))

        return hits(ranks)

    half_a = [q for q in halves["A"] if q in data]
    half_b = [q for q in halves["B"] if q in data]
    everything = half_a + half_b
    zero = np.zeros(len(FEATURES))

    best, best_key = zero, (-1.0, -1.0, -1.0)
    for combo in itertools.product(GRID, repeat=len(FEATURES)):
        result = evaluate(half_a, np.array(combo))
        key = (result["top5"], result["top10"], result["top1"])

        if key > best_key:
            best, best_key = np.array(combo), key

    # Each feature alone at weight 1, on all 200, to see which carries signal.
    single = {
        name: evaluate(everything, np.eye(len(FEATURES))[index])
        for index, name in enumerate(FEATURES)
    }

    report = {
        "weights (picked on half A)": dict(zip(FEATURES, best.tolist(), strict=True)),
        "reranker only": {
            "half A": evaluate(half_a, zero),
            "half B": evaluate(half_b, zero),
            "all": evaluate(everything, zero),
        },
        "reranker + fields": {
            "half A": evaluate(half_a, best),
            "half B": evaluate(half_b, best),
            "all": evaluate(everything, best),
        },
        "one feature at weight 1 (all)": single,
    }

    RESULT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
