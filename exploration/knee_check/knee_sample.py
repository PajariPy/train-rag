"""Does the biggest-drop rule find the right document on ordinary questions?

On the 100-question sample of `exploration/embedder_comparison/` (stratified by type), the
reranker scores each question's fused top 50 exactly as the pipeline does (title + chunk
text, normalized query). Then, per question: find the biggest drop between neighbours in
the top 30, and check whether the right document is above it.

Two settings:
- as production sees it: the fused 50 only (a right document missing from it counts as
  "not in the list");
- as training would see it: the right document added to the list when missing, like a
  synthetic question whose source document is always there.

Free, local, about 5 minutes. Scores are cached in data/.

Usage:
    uv run python exploration/knee_check/knee_sample.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sqlalchemy import text

from train_rag.database import get_engine
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress
from train_rag.reranker import GteModernBertReranker

ROOT = Path(__file__).parent.parent
SAMPLE = ROOT / "embedder_comparison/data/comparison.json"
REPLAY = ROOT / "retriever_attribution/data/attribution.json"
DATA = Path(__file__).parent / "data"
SCORES = DATA / "rerank_scores.json"
KNEE_WINDOW = 30


def load_texts(doc_ids: list[str]) -> dict[str, str]:
    """Title + text of each document, as the pipeline feeds the reranker.

    Args:
        doc_ids: The documents.

    Returns:
        doc_id -> the title, a line break and the text.
    """
    with get_engine().connect() as connection:
        rows = connection.execute(
            text("select doc_id, title, content from documents where doc_id = any(:ids)"),
            {"ids": doc_ids},
        ).all()

    return {row.doc_id: f"{row.title}\n{row.content}" for row in rows}


def biggest_drop(scores: list[float]) -> int:
    """Number of documents above the biggest drop between neighbours in the top window.

    Args:
        scores: Scores sorted from high to low.

    Returns:
        How many documents sit above the drop.
    """
    top = np.array(scores[: KNEE_WINDOW + 1])

    return int(np.argmax(top[:-1] - top[1:])) + 1


def main() -> None:
    """Score the sample with the reranker, apply the rule in both settings, print the split."""
    questions = json.loads(SAMPLE.read_text())["questions"]
    replay = json.loads(REPLAY.read_text())
    cached = json.loads(SCORES.read_text()) if SCORES.exists() else {}
    todo = [question for question in questions if question["id"] not in cached]
    if todo:
        reranker = GteModernBertReranker()
        DATA.mkdir(parents=True, exist_ok=True)
        with RunProgress(
            get_engine(), "rerank 100-question sample (free)", len(todo), unit="questions"
        ) as progress:
            for question in todo:
                doc_ids = list(
                    dict.fromkeys(replay[question["id"]]["fused"] + question["expected"])
                )
                texts = load_texts(doc_ids)
                doc_ids = [doc_id for doc_id in doc_ids if doc_id in texts]
                scores = reranker.score(
                    normalize_text(question["text"]), [texts[doc_id] for doc_id in doc_ids]
                )
                cached[question["id"]] = dict(zip(doc_ids, scores, strict=True))
                SCORES.write_text(json.dumps(cached))
                progress.advance(question["type"], 1, note="local, $0")

    for setting in ("production (fused 50 only)", "training (right document always in the list)"):
        outcome: Counter[str] = Counter()
        by_type: dict[str, Counter[str]] = {}
        near = []
        for question in questions:
            scores = cached[question["id"]]
            fused = replay[question["id"]]["fused"]
            pool = (
                scores
                if setting.startswith("training")
                else {d: s for d, s in scores.items() if d in fused}
            )
            order = sorted(pool, key=lambda doc_id: -pool[doc_id])
            right = [order.index(doc_id) + 1 for doc_id in question["expected"] if doc_id in pool]
            if not right:
                result = "not in the list"
            else:
                cluster = biggest_drop([pool[doc_id] for doc_id in order])
                rank = min(right)
                result = "above the drop" if rank <= cluster else "below the drop"
                if result == "below the drop":
                    near.append(rank - cluster)
            outcome[result] += 1
            by_type.setdefault(question["type"], Counter())[result] += 1
        print(f"\n{setting}: {dict(outcome)}")
        print(f"  below the drop, ranks under it: {sorted(near)}")
        print(f"  within 3 ranks under the drop: {sum(gap <= 3 for gap in near)}")
        for question_type, counts in sorted(by_type.items()):
            print(
                f"  {question_type:25s} above {counts['above the drop']:2d}  "
                f"below {counts['below the drop']:2d}"
                f"  not in list {counts['not in the list']:2d}"
            )


if __name__ == "__main__":
    main()
