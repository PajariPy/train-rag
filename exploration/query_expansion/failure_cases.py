"""The semantic benchmark questions today's best setup still fails, with what each search found.

Best setup (16 Nov decision): fine-tuned gte on the question, BM25 on the question with the
4 Gemini rewrites appended, the union of both top 50 reranked. A question fails when none of
its right documents reaches the reranked top 5.

For every failure: the question, the gold answer, each right document (its rank in the gte
and BM25 lists and in the reranked union, and the passage closest to the gold answer), and
the top documents of gte, BM25 and the reranker, each with the passage closest to the gold
answer, to compare what they share with the right document and how they differ.

Reads data/benchmark_merge_check.json (benchmark_merge_check.py); local, free. Writes
data/failure_cases.json and data/failure_cases.md (for reading).

Usage:
    uv run python exploration/query_expansion/failure_cases.py
"""

import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy import bindparam, text

from train_rag.database import get_engine

DATA = Path(__file__).parent / "data"
RESULTS = DATA / "benchmark_merge_check.json"
OUTPUT = DATA / "failure_cases.json"
READABLE = DATA / "failure_cases.md"
GTE, BM25 = "gte:q", "bm25:concat"
SHOWN = 3  # top documents shown per list
PASSAGE = 320  # characters of the passage closest to the gold answer
WORD = re.compile(r"[a-z0-9][a-z0-9_.\-]*")


def words(value: str) -> set[str]:
    """Lowercase words longer than three characters."""
    return {word for word in WORD.findall(value.lower()) if len(word) > 3}


def closest_passage(content: str, answer: str) -> str:
    """The window of a document sharing the most words with the gold answer."""
    wanted = words(answer)
    best, best_overlap = content[:PASSAGE], -1
    for start in range(0, max(1, len(content) - PASSAGE // 2), PASSAGE // 4):
        window = content[start : start + PASSAGE]
        overlap = len(wanted & words(window))
        if overlap > best_overlap:
            best, best_overlap = window, overlap

    return " ".join(best.split())


def load_questions() -> dict[str, dict[str, Any]]:
    """The semantic benchmark questions with their gold answers and right documents."""
    with get_engine().connect() as connection:
        rows = connection.execute(
            text(
                "select id, text, gold_answer, expected_doc_ids from questions "
                "where question_type = 'semantic' order by id"
            )
        ).all()

    return {
        row.id: {"text": row.text, "gold": row.gold_answer, "expected": row.expected_doc_ids}
        for row in rows
    }


def load_documents(doc_ids: set[str]) -> dict[str, dict[str, str]]:
    """Source, first title line and content of some documents."""
    query = text(
        "select doc_id, source_type, title, content from documents where doc_id in :ids"
    ).bindparams(bindparam("ids", expanding=True))
    with get_engine().connect() as connection:
        rows = connection.execute(query, {"ids": sorted(doc_ids)}).all()

    return {
        row.doc_id: {
            "source": row.source_type,
            "title": (row.title or "").split("\n")[0][:110],
            "content": row.content or "",
        }
        for row in rows
    }


def rank(doc_id: str, ranking: list[str]) -> int | None:
    """1-based position of a document in a list, or None."""
    return ranking.index(doc_id) + 1 if doc_id in ranking else None


def main() -> None:
    """Collect the failures, write them as JSON and as a readable Markdown file."""
    questions = load_questions()
    results = json.loads(RESULTS.read_text())
    cases = []
    for question_id, result in results.items():
        lists, scores = result["lists"], result["scores"]
        pool = set(lists[GTE]) | set(lists[BM25])
        reranked = sorted(pool, key=lambda doc_id: scores[doc_id], reverse=True)
        expected = questions[question_id]["expected"]
        if set(expected) & set(reranked[:5]):
            continue
        cases.append(
            {
                "id": question_id,
                "lost_by": "reranker" if set(expected) & pool else "search",
                "gte": lists[GTE][:SHOWN],
                "bm25": lists[BM25][:SHOWN],
                "reranked": reranked[:SHOWN],
                "expected": [
                    {
                        "doc_id": doc_id,
                        "gte_rank": rank(doc_id, lists[GTE]),
                        "bm25_rank": rank(doc_id, lists[BM25]),
                        "rerank_rank": rank(doc_id, reranked),
                        "score": scores.get(doc_id),
                    }
                    for doc_id in expected
                ],
                "scores": {doc_id: scores[doc_id] for doc_id in reranked[:SHOWN]},
            }
        )

    shown = {
        doc_id
        for case in cases
        for doc_id in [*case["gte"], *case["bm25"], *case["reranked"]]
        + [e["doc_id"] for e in case["expected"]]
    }
    documents = load_documents(shown)
    for case in cases:
        question = questions[case["id"]]
        case |= {"question": question["text"], "gold": question["gold"]}
        case["documents"] = {
            doc_id: {
                "source": documents[doc_id]["source"],
                "title": documents[doc_id]["title"],
                "passage": closest_passage(documents[doc_id]["content"], question["gold"]),
                "chars": len(documents[doc_id]["content"]),
            }
            for doc_id in [*case["gte"], *case["bm25"], *case["reranked"]]
            + [e["doc_id"] for e in case["expected"]]
            if doc_id in documents
        }
    OUTPUT.write_text(json.dumps(cases, indent=1))
    READABLE.write_text(readable(cases))
    lost = sum(case["lost_by"] == "reranker" for case in cases)
    print(f"{len(cases)} failures: {len(cases) - lost} not in the candidates, {lost} lost by the reranker")


def readable(cases: list[dict[str, Any]]) -> str:
    """The failures as Markdown, one section per question."""
    lines = []
    for case in sorted(cases, key=lambda c: (c["lost_by"], c["id"])):
        docs = case["documents"]
        lines += [f"## {case['id']} ({case['lost_by']})", f"Q: {case['question']}", f"Gold: {case['gold']}"]
        for e in case["expected"]:
            doc = docs.get(e["doc_id"], {})
            lines += [
                f"RIGHT {e['doc_id']} [{doc.get('source')}] {doc.get('title')} ({doc.get('chars')} chars)"
                f" gte#{e['gte_rank']} bm25#{e['bm25_rank']} rerank#{e['rerank_rank']}"
                f" score {e['score'] if e['score'] is None else round(e['score'], 3)}",
                f"  > {doc.get('passage')}",
            ]
        for name in ("reranked", "gte", "bm25"):
            for i, doc_id in enumerate(case[name], 1):
                doc = docs[doc_id]
                score = case["scores"].get(doc_id)
                tag = f" {score:.3f}" if score is not None else ""
                lines += [
                    f"{name} {i}: {doc_id} [{doc['source']}] {doc['title']}{tag}",
                    f"  > {doc['passage']}" if name == "reranked" or doc_id not in case["reranked"] else "  > (as above)",
                ]
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
