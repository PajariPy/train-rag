"""Build the dev-contradictions set: consistent, specific and ambiguous questions, equal shares.

Starts from the ambiguity run (ambiguity.py): for each of the 200 Slack dev questions, the
right thread and every document of today's and agent v3's final top 5, each rated alone with
the question (full / partial / none, with its answer and quote, cached in
exploration/question_ambiguity/rating_cache.json).

1. Compare: for every other document that answers fully or partly, Gemini compares its answer
   with the right thread's: same case and consistent, same case but conflicting facts, or a
   different case.
2. Classes:
   - consistent: no other shown document answers, or all that do are the same case with
     consistent facts. The answer doesn't depend on which document is picked.
   - ambiguous: at least one other document answers fully and is a different case or
     conflicts. The question needs a clarifying question (or the cases side by side).
   - specific: made from ambiguous questions not used in the ambiguous class. Gemini adds one
     detail the person would plausibly know (where it was discussed, which customer or team,
     roughly when, which environment), never the answer. Kept only if the right thread still
     answers fully and every conflicting document now rates "none".
3. Equal shares (fixed seed): N = the smallest class, at most MAX_PER_CLASS.

Writes data/dev_contradictions.jsonl (one question per line: id, class, question, original
question, right doc_id, the conflicting and consistent doc_ids with their answers) and a
summary in data/dev_contradictions_report.json.

Paid: Gemini (cached). Usage:
    uv run python exploration/slack_extraction/contradictions.py
"""

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from dotenv import dotenv_values
from field_match import HERE, ROOT
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel
from sqlmodel import Session

from train_rag.database import get_engine
from train_rag.evaluation.judge import GEMINI_MODEL
from train_rag.evaluation.usage import UsageMeter
from train_rag.processing.normalize import normalize_content
from train_rag.repository.document import ids_of_doc_ids, load_documents

sys.path.insert(0, str(ROOT / "exploration" / "question_ambiguity"))
from answers import DocumentAnswer, RatingCache, rate  # noqa: E402
from llm_cache import JsonCache, hash_key  # noqa: E402

AMBIGUITY = HERE / "data" / "ambiguity_result.json"
QUESTIONS = ROOT / "exploration" / "training_questions" / "data" / "questions.jsonl"
CACHE = HERE / "data" / "contradictions_cache.json"
OUTPUT = HERE / "data" / "dev_contradictions.jsonl"
REPORT = HERE / "data" / "dev_contradictions_report.json"
SEED = 23
MAX_PER_CLASS = 50
WORKERS = 8

COMPARE = """You compare two answers to the same question, each taken from a different internal
company document (chat threads, emails, tickets, docs).

First, in reasoning, say in two or three sentences what case or event each answer is about
(which incident, customer, release, decision, time) and whether their facts agree.

Then set relation:
- same_case_consistent: both are about the same case and their facts agree (one may give
  fewer details);
- same_case_conflicting: both are about the same case but state different facts (another
  fix, owner, number, date or decision);
- different_case: they are about different cases, events or customers that both fit the
  question's words.
"""

SPECIFY = """A search question fits several company documents that describe different cases. You
make the question specific to one document by adding one detail, the way the person who
asked could have said it.

Add exactly one short detail that the target document has and the other documents don't:
where it was discussed (a channel, an email, a ticket), which customer, team or
environment, or roughly when. The person knows the context of their question, not its
answer: never add the answer, the fix, the decision, numbers, ids or people's names.
Keep the rest of the question as it is, and keep it natural.

Reply as JSON: "detail" (the detail you added) and "question" (the new question).
"""


class Comparison(BaseModel):
    """How another document's answer relates to the right thread's."""

    reasoning: str
    relation: Literal["same_case_consistent", "same_case_conflicting", "different_case"]


class Specific(BaseModel):
    """The question with one distinguishing detail added."""

    detail: str
    question: str


def call(
    client: genai.Client, usage: UsageMeter, cache: JsonCache, system: str, text: str,
    schema: type[BaseModel],
) -> BaseModel:
    """One cached Gemini call with a JSON schema."""
    key = hash_key([GEMINI_MODEL, system, text])
    if (hit := cache.get(key)) is not None:
        return schema.model_validate(hit)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        ),
    )
    usage.add(response)
    result = schema.model_validate_json(response.text or "")
    cache.put(key, result.model_dump())
    return result


def main() -> None:
    """Compare answers, rewrite specific questions, balance the classes and write the set."""
    env = dotenv_values(ROOT / ".env")
    ambiguity = json.loads(AMBIGUITY.read_text())["questions"]
    questions = {
        row["doc_id"]: row["question"]
        for row in map(json.loads, QUESTIONS.open())
        if row["split"] == "dev_slack"
    }
    shown = {
        q: sorted({d for d, _ in v["today"] + v["agent"]} - {q}) for q, v in ambiguity.items()
    }
    coverage = {
        (q, d): c for q, v in ambiguity.items() for d, c in v["today"] + v["agent"]
    } | {(q, q): v["right"] for q, v in ambiguity.items()}

    doc_ids = sorted({d for q in shown for d in [q, *shown[q]]})
    with Session(get_engine()) as session:
        documents = {
            doc.doc_id: doc
            for doc in load_documents(
                session, list(ids_of_doc_ids(session, "enterprise_rag_bench", doc_ids))
            )
        }

    client = genai.Client(api_key=env["GEMINI_API_KEY"])
    usage, ratings, cache = UsageMeter(), RatingCache(), JsonCache(CACHE)

    def answer(q: str, d: str, question: str | None = None) -> DocumentAnswer:
        rated = rate(client, usage, ratings, question or questions[q], documents[d])
        if rated is None:
            raise ValueError(f"no rating for {d}")
        return rated

    # 1. Compare every answering document with the right thread.
    pairs = [
        (q, d) for q in ambiguity if ambiguity[q]["right"] == "full"
        for d in shown[q] if coverage.get((q, d)) in ("full", "partial") and d in documents
    ]
    print(f"{len(pairs)} answers to compare", flush=True)

    def compare(pair: tuple[str, str]) -> tuple[tuple[str, str], str | None]:
        q, d = pair
        try:
            right, other = answer(q, q), answer(q, d)
            text = (
                f"Question: {questions[q]}\n\nAnswer A (document 1): {right.answer}\n"
                f"Passage A: {right.quote}\n\nAnswer B (document 2): {other.answer}\n"
                f"Passage B: {other.quote}"
            )
            return pair, call(client, usage, cache, COMPARE, text, Comparison).relation
        except (genai_errors.APIError, ValueError):
            return pair, None

    with ThreadPoolExecutor(WORKERS) as pool:
        relation = dict(pool.map(compare, pairs))
    cache.save()
    ratings.save()

    # 2. Classes from the comparisons.
    consistent, ambiguous = [], []
    conflicts: dict[str, list[str]] = {}
    agreeing: dict[str, list[str]] = {}
    for q in ambiguity:
        if ambiguity[q]["right"] != "full":
            continue
        mine = [(d, relation.get((q, d))) for d in shown[q] if (q, d) in relation]
        if any(r is None for _, r in mine):
            continue
        conflicts[q] = [
            d for d, r in mine if r != "same_case_consistent" and coverage[(q, d)] == "full"
        ]
        agreeing[q] = [d for d, r in mine if r == "same_case_consistent"]
        partial_conflict = any(
            r != "same_case_consistent" and coverage[(q, d)] == "partial" for d, r in mine
        )
        if conflicts[q]:
            ambiguous.append(q)
        elif not partial_conflict:
            consistent.append(q)

    rng = random.Random(SEED)
    rng.shuffle(ambiguous)
    rng.shuffle(consistent)
    # Consistent questions are the scarce class: keep that many ambiguous ones and try to make
    # all the rest specific.
    target = min(len(consistent), MAX_PER_CLASS)
    keep_ambiguous, to_specify = ambiguous[:target], ambiguous[target:]

    # 3. Specific: one distinguishing detail, verified by re-rating.
    def specify(q: str) -> tuple[str, dict[str, Any] | None]:
        try:
            others = "\n\n".join(
                f"Other document ({documents[d].source_type}): {answer(q, d).answer}"
                for d in conflicts[q]
            )
            target = normalize_content(documents[q].content)[:6000]
            text = (
                f"Question: {questions[q]}\n\nTarget document ({documents[q].source_type}: "
                f"{documents[q].title}):\n{target}\n\n{others}"
            )
            # Up to three attempts; each retry is told what was wrong with the last version.
            for _attempt in range(3):
                new = call(client, usage, cache, SPECIFY, text, Specific)
                if re.search(r"\d", new.detail):  # ids, versions, dates: not what people recall
                    text += f"\n\nYour detail \"{new.detail}\" has an id or number; use context."
                    continue
                if answer(q, q, new.question).coverage != "full":
                    text += f"\n\nYour version \"{new.question}\" no longer fits the target."
                    continue
                still = [d for d in conflicts[q] if answer(q, d, new.question).coverage != "none"]
                if not still:
                    return q, {"question": new.question, "detail": new.detail}
                text += (
                    f"\n\nYour version \"{new.question}\" still fits "
                    f"{len(still)} other document(s); pick a detail that rules them out."
                )
            return q, None
        except (genai_errors.APIError, ValueError):
            return q, None

    with ThreadPoolExecutor(WORKERS) as pool:
        specified = {q: s for q, s in pool.map(specify, to_specify) if s}
    cache.save()
    ratings.save()

    # 4. Equal shares and the output.
    n = min(len(consistent), len(keep_ambiguous), len(specified), MAX_PER_CLASS)
    chosen = (
        [("consistent", q) for q in consistent[:n]]
        + [("ambiguous", q) for q in keep_ambiguous[:n]]
        + [("specific", q) for q in list(specified)[:n]]
    )

    def row(kind: str, q: str) -> dict[str, Any]:
        text = specified[q]["question"] if kind == "specific" else questions[q]
        return {
            "id": f"{kind}:{q}", "class": kind, "question": text,
            "original_question": questions[q], "right_doc_id": q,
            "added_detail": specified[q]["detail"] if kind == "specific" else None,
            "right_answer": answer(q, q).answer,
            "conflicting": [
                {"doc_id": d, "answer": answer(q, d).answer, "relation": relation[(q, d)]}
                for d in conflicts.get(q, [])
            ],
            "consistent": [
                {"doc_id": d, "answer": answer(q, d).answer} for d in agreeing.get(q, [])
            ],
        }

    OUTPUT.write_text("".join(json.dumps(row(k, q), ensure_ascii=False) + "\n" for k, q in chosen))
    report = {
        "right thread full": sum(v["right"] == "full" for v in ambiguity.values()),
        "answers compared": len(pairs),
        "relations": {r: list(relation.values()).count(r) for r in set(relation.values())},
        "consistent available": len(consistent),
        "ambiguous available": len(ambiguous),
        "ambiguous kept as ambiguous": len(keep_ambiguous),
        "specific tried": len(to_specify),
        "specific verified": len(specified),
        "per class": n,
        "gemini calls": usage.calls,
    }
    REPORT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
