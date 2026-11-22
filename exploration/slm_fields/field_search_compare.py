"""Level 1 search test: does field search find the right thread as well with the SLM's fields?

The "Fields as a search" test of article 007 (`slack_extraction/field_pool.py`), run twice on
the same questions and pool, once per extractor:

- questions: half B of the 200 Slack dev questions (never trained on);
- pool: every held-out document (`test` + `extraction_test`, ~12.5k), the 100 right threads
  included; half A's candidates are left out because the SLM was trained on them;
- extractors: DeepSeek's labels (prompt fc8ae8878988, as used in the field test; profile
  vectors from its cache) and the SLM run's outputs (runs/<name>/outputs.jsonl; an output
  that doesn't parse counts as empty fields). Each side uses its own fields for the
  question and for the documents.

Each document is ranked by gte similarity between the question's and the document's profiles
(subject = topic + concepts, problem = bug + cause + fix + question, names), with the
mixes of field_pool.py, next to today's dense search on the same pool (fine-tuned gte, best
stored chunk). Reported for the questions whose right thread today's search reaches and for
those it misses: right thread in the top 10 / 50 / 100 / 300, median rank.

Free and local (gte on this machine). Writes runs/<name>/field_search.json and .md.
Usage: uv run python exploration/slm_fields/field_search_compare.py --name fields-r16-e2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
EXTRACTION = HERE.parent / "slack_extraction"
sys.path.insert(0, str(EXTRACTION))
from field_match import BASELINE, CACHE, HALVES, NAMES, load_labels  # noqa: E402
from field_pool import MIXES, raw_gte_scores  # noqa: E402
from schema import FIELDS  # noqa: E402

from train_rag.database import get_engine  # noqa: E402
from train_rag.embedder import FineTunedGteEmbedder  # noqa: E402
from train_rag.progress.recorder import RunProgress  # noqa: E402

DEPTHS = (10, 50, 100, 300)
BATCH = 2_048


def profiles(fields: dict[str, Any]) -> tuple[str, str, str]:
    """Subject, problem and names strings, as field_match.profiles, tolerant of bad output."""

    def values(name: str) -> list[str]:
        found = fields.get(name, []) if isinstance(fields, dict) else []
        return [str(v) for v in (found if isinstance(found, list) else [found]) if v]

    subject = "; ".join(values("topic") + values("concept"))
    problem = "; ".join(values("bug") + values("cause") + values("fix") + values("question"))
    names = "; ".join(v for field in NAMES for v in values(field))

    return subject, problem, names


def slm_fields(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """The SLM's fields per item id, and how many outputs didn't parse (kept as empty)."""
    fields, broken = {}, 0
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            got = json.loads(row["text"])
            fields[row["doc_id"]] = got.get("fields", {}) if isinstance(got, dict) else {}
        except json.JSONDecodeError:
            fields[row["doc_id"]], broken = {}, broken + 1

    return fields, broken


def embed(ids: list[str], fields: dict[str, dict[str, Any]], cache: Path) -> dict[str, np.ndarray]:
    """Unit gte vectors (subject, problem, names) per id; empty profiles get zero vectors."""
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if list(stored["ids"]) == ids:
            return dict(zip(ids, stored["vectors"], strict=True))
    texts = [text for item in ids for text in profiles(fields[item])]
    embedder = FineTunedGteEmbedder(batch_size=256)
    chunks = []
    with RunProgress(get_engine(), "embed SLM field profiles (gte)", len(texts)) as progress:
        for start in range(0, len(texts), BATCH):
            batch = texts[start : start + BATCH]
            chunks.append(np.asarray(embedder.embed([t or "none" for t in batch]), np.float32))
            progress.advance("embedding", len(batch))
    vectors = np.concatenate(chunks)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    vectors = vectors.reshape(len(ids), 3, -1)
    for row, text in enumerate(texts):
        if not text:
            vectors[row // 3, row % 3] = 0
    np.savez(cache, ids=np.array(ids), vectors=vectors)

    return dict(zip(ids, vectors, strict=True))


def ranks(
    vectors: dict[str, np.ndarray], pool: list[str], groups: dict[str, list[str]],
    weights: tuple[float, float, float] | None, dense: np.ndarray, row_of: dict[str, int],
) -> dict[str, list[int]]:
    """The right thread's rank per question, by group, for one mix (None = today's dense)."""
    matrix = np.stack([vectors[item] for item in pool])
    position = {item: index for index, item in enumerate(pool)}

    def z(values: np.ndarray) -> np.ndarray:
        return (values - values.mean()) / (values.std() + 1e-9)

    result: dict[str, list[int]] = {}
    for group, questions in groups.items():
        result[group] = []
        for doc_id in questions:
            if weights is None:
                scores = dense[row_of[doc_id]]
            else:
                q_vec = vectors[f"question:{doc_id}"]
                scores = sum(w * (matrix[:, c] @ q_vec[c]) for c, w in enumerate(weights) if w)
            result[group].append(int((scores > scores[position[doc_id]]).sum()) + 1)

    return result


def hybrid_ranks(
    baseline: dict[str, dict[str, Any]], groups: dict[str, list[str]], pool: set[str] | None
) -> dict[str, list[float]]:
    """The right thread's rank in today's hybrid search, by group (inf = not in its list).

    Hybrid = fine-tuned gte top 50 on the question + BM25 top 50 on the question with the 4
    rewrites appended, the union ordered by the reranker (16 Nov decision), as stored by
    query_expansion/slack_dev_baseline.py. With `pool`, documents outside it are dropped
    from the list first (the same ground as field search).
    """
    result: dict[str, list[float]] = {}
    for group, questions in groups.items():
        result[group] = []
        for doc_id in questions:
            entry = baseline[doc_id]
            union = set(entry["lists"]["gte:q"]) | set(entry["lists"]["bm25:concat"])
            if pool is not None:
                union &= pool
            order = sorted(union, key=lambda item: entry["scores"][item], reverse=True)
            result[group].append(order.index(doc_id) + 1 if doc_id in order else float("inf"))

    return result


def summary(values: list[float]) -> dict[str, float]:
    """Share of questions with the right thread in each top k, and the median rank."""
    array = np.array(values)

    median = float(np.median(array))

    return {f"top{d}": round(100 * float(np.mean(array <= d)), 1) for d in DEPTHS} | {
        "median rank": int(median) if np.isfinite(median) else "not found"
    }


def main() -> None:
    """Rank the pool with each extractor's fields and write the comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run = HERE / "runs" / args.name

    slm, broken = slm_fields(run / "outputs.jsonl")
    baseline: dict[str, dict[str, Any]] = json.loads(BASELINE.read_text())
    half_b = json.loads(HALVES.read_text())["B"]
    pool = sorted(item for item in slm if not item.startswith("question:"))
    pool_set = set(pool)
    questions = [q for q in half_b if q in pool_set and f"question:{q}" in slm]
    groups = {
        "reached": [q for q in questions if q in baseline[q]["scores"]],
        "missed": [q for q in questions if q not in baseline[q]["scores"]],
    }
    ids = pool + [f"question:{q}" for q in questions]

    deepseek_labels = load_labels()
    stored = np.load(CACHE, allow_pickle=False)  # the field test's DeepSeek vectors (17 Nov)
    cached = dict(zip(stored["ids"], stored["vectors"], strict=True))
    missing = [item for item in ids if item not in cached]
    if missing:
        deepseek_fields = {item: deepseek_labels[item]["fields"] for item in missing}
        cached |= embed(missing, deepseek_fields, run / "deepseek_extra_vectors.npz")
    deepseek = {item: cached[item] for item in ids}
    slm_vectors = embed(ids, {item: slm[item] for item in ids}, run / "slm_profile_vectors.npz")

    rows = [
        json.loads(line)
        for line in (HERE.parents[0] / "training_questions" / "data" / "questions.jsonl").open()
    ]
    texts = {row["doc_id"]: row["question"] for row in rows if row["split"] == "dev_slack"}
    order = groups["reached"] + groups["missed"]
    dense = raw_gte_scores(pool, order, texts)
    row_of = {q: index for index, q in enumerate(order)}

    report: dict[str, Any] = {
        "pool": len(pool), "questions": len(questions),
        "reached": len(groups["reached"]), "missed": len(groups["missed"]),
        "slm_outputs_not_parsed": broken, "deepseek_vectors_from_cache": len(ids) - len(missing),
        "today: raw gte, same pool": {
            g: summary(v) for g, v in ranks(deepseek, pool, groups, None, dense, row_of).items()
        },
    }
    for label, restrict in (("hybrid, whole corpus", None), ("hybrid, same pool", pool_set)):
        found = hybrid_ranks(baseline, groups, restrict)
        report[f"today: {label}"] = {g: summary(v) for g, v in found.items()} | {
            "all": summary(found["reached"] + found["missed"])
        }
    for extractor, vectors in (("DeepSeek", deepseek), ("SLM", slm_vectors)):
        for mix, weights in MIXES.items():
            found = ranks(vectors, pool, groups, weights, dense, row_of)
            report[f"{extractor}: {mix}"] = {g: summary(v) for g, v in found.items()} | {
                "all": summary(found["reached"] + found["missed"])
            }
    (run / "field_search.json").write_text(json.dumps(report, indent=1))

    lines = [
        f"# Field search, half B ({report['questions']} questions, pool {report['pool']:,})",
        "", f"Reached by today's search: {report['reached']}; missed: {report['missed']}. "
        f"SLM outputs that didn't parse: {broken}.", "",
        "| Method | Reached: top 10 / 50 / 100 / 300 | Missed: top 10 / 50 / 100 / 300 "
        "| All: top 10 / 50 / 100 | Median rank (all) |",
        "|---|---|---|---|---:|",
    ]
    for method, result in report.items():
        if not isinstance(result, dict):
            continue

        def cells(r: dict[str, float], depths: tuple[int, ...] = DEPTHS) -> str:
            return " / ".join(f"{r[f'top{d}']:.0f}" for d in depths)

        every = result.get("all")
        lines.append(
            f"| {method} | {cells(result['reached'])} | {cells(result['missed'])} "
            f"| {cells(every, (10, 50, 100)) if every else '–'} "
            f"| {every['median rank'] if every else '–'} |"
        )
    text = "\n".join(lines) + "\n"
    (run / "field_search.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
