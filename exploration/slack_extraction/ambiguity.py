"""Ambiguity of the 200 Slack dev questions: what does each shown document answer on its own?

The method of exploration/question_ambiguity (answers.py, same prompt and cache): Gemini
reads one document alone with the question and says how fully it answers (full / partial /
none), with the answer and the passage it rests on. Rated per question:

- the right thread (the one the question was written from): can the question be answered
  from it at all, without extra details?
- the final top 5 after the reranker of two setups: today's candidates, and agent v3's top
  100 (agent_rerank.py).

Reported:
- right thread: full / partial / none (none or partial = the question asks more than its
  own thread says);
- ambiguous: a document other than the right thread answers fully (or partly);
- "answer found": the top 1 / top 5 holds a document that answers fully, any document, next
  to the strict "right thread in top 1 / 5".

Paid: Gemini (cached per question and document, like answers.py). Usage:
    uv run python exploration/slack_extraction/ambiguity.py
"""

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent_rerank import AGENT, SCORES, agent_lists
from agent_search import QUESTIONS, FieldIndex
from dotenv import dotenv_values
from field_match import BASELINE, HERE, ROOT
from google import genai
from google.genai import errors as genai_errors
from sqlmodel import Session

from train_rag.database import get_engine
from train_rag.evaluation.usage import UsageMeter
from train_rag.repository.document import ids_of_doc_ids, load_documents

sys.path.insert(0, str(ROOT / "exploration" / "question_ambiguity"))
from answers import RatingCache, rate  # noqa: E402

RESULT = HERE / "data" / "ambiguity_result.json"
TOP = 5
WORKERS = 8


def main() -> None:
    """Rate the right thread and both setups' top 5 for every question, then report."""
    env = dotenv_values(ROOT / ".env")
    baseline: dict[str, dict[str, Any]] = json.loads(BASELINE.read_text())
    rows = [json.loads(line) for line in QUESTIONS.open()]
    questions = {row["doc_id"]: row["question"] for row in rows if row["split"] == "dev_slack"}
    runs = [r for r in json.loads(AGENT.read_text())["runs"] if "error" not in r]
    lists = agent_lists(FieldIndex(), runs)
    cached = json.loads(SCORES.read_text())

    tops: dict[str, dict[str, list[str]]] = {}
    for q in lists:
        scores = dict(baseline[q]["scores"]) | cached.get(q, {})
        today = sorted(baseline[q]["scores"], key=lambda d: -scores[d])[:TOP]
        agent = sorted(lists[q][:100], key=lambda d: -scores.get(d, -1e9))[:TOP]
        tops[q] = {"today": today, "agent": agent}

    pairs = sorted({(q, d) for q, t in tops.items() for d in [q, *t["today"], *t["agent"]]})
    doc_ids = sorted({d for _, d in pairs})
    with Session(get_engine()) as session:
        documents = {
            doc.doc_id: doc
            for doc in load_documents(session, list(ids_of_doc_ids(session, "enterprise_rag_bench", doc_ids)))
        }

    client = genai.Client(api_key=env["GEMINI_API_KEY"])
    usage = UsageMeter()
    cache = RatingCache()
    print(f"{len(pairs):,} (question, document) pairs to rate", flush=True)

    def one(pair: tuple[str, str]) -> tuple[tuple[str, str], str | None]:
        q, d = pair
        if d not in documents:
            return pair, None
        try:
            answer = rate(client, usage, cache, questions[q], documents[d])
        except (genai_errors.APIError, ValueError):
            return pair, None
        return pair, answer.coverage if answer else None

    with ThreadPoolExecutor(WORKERS) as pool:
        coverage = dict(pool.map(one, pairs))
    cache.save()

    failed = sum(1 for v in coverage.values() if v is None)
    right = Counter(coverage[(q, q)] for q in tops)

    def setup(name: str) -> dict[str, Any]:
        found = {k: 0 for k in ("right top1", "right top5", "answer top1", "answer top5",
                                "partial-or-full top5")}
        for q, t in tops.items():
            top = t[name]
            found["right top1"] += top[:1] == [q]
            found["right top5"] += q in top
            found["answer top1"] += bool(top) and coverage.get((q, top[0])) == "full"
            found["answer top5"] += any(coverage.get((q, d)) == "full" for d in top)
            found["partial-or-full top5"] += any(coverage.get((q, d)) in ("full", "partial") for d in top)
        return {k: round(100 * v / len(tops), 1) for k, v in found.items()}

    other_full = sum(
        any(coverage.get((q, d)) == "full" for d in {*t["today"], *t["agent"]} - {q})
        for q, t in tops.items()
    )
    other_any = sum(
        any(coverage.get((q, d)) in ("full", "partial") for d in {*t["today"], *t["agent"]} - {q})
        for q, t in tops.items()
    )
    report = {
        "questions": len(tops), "pairs rated": len(pairs) - failed, "failed": failed,
        "gemini calls": usage.calls,
        "right thread answers": dict(right),
        "ambiguous: another shown document answers fully": round(100 * other_full / len(tops), 1),
        "another shown document answers fully or partly": round(100 * other_any / len(tops), 1),
        "today": setup("today"),
        "agent 100": setup("agent"),
    }
    per_question = {
        q: {"right": coverage[(q, q)], "today": [[d, coverage.get((q, d))] for d in t["today"]],
            "agent": [[d, coverage.get((q, d))] for d in t["agent"]]}
        for q, t in tops.items()
    }
    RESULT.write_text(json.dumps({"report": report, "questions": per_question}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
