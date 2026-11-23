"""Evaluate the agreement classifier on the dev-contradictions set.

The classifier is the first LLM step of the disagreement design (docs/todo/Show the agent
where its results disagree.md): given a question and the documents search returned, does
the answer depend on which document is picked? It never answers the question itself.

Each of the 84 questions in data/dev_contradictions.jsonl (28 consistent, 28 specific,
28 ambiguous, built by contradictions.py) gets the documents the ambiguity run showed for
it (today's and agent v3's final top 5) plus the right thread, in a shuffled order (fixed
seed), each as its title, source and the start of its text. The classifier says:

- same_answer: every document that answers says the same thing (class consistent);
- one_case_fits: the documents describe different cases, but the question singles out
  one (class specific);
- several_cases_fit: different cases fit the question as asked, so the person has to be
  asked which one (class ambiguous);

plus the ids of the documents that fit the question as asked.

Scores: the 3x3 confusion matrix, the ask decision (several_cases_fit vs the rest), and
whether the right thread is among the documents it says fit (for specific: without the
conflicting ones).

Paid: Gemini, one call per question (cached). Usage, from the repo root:
    uv run python exploration/slack_extraction/agreement.py [version] [full|head|snippet] [repeat k]

Labels: the class after the hand review (data/dev_contradictions_review.json, applied to the
set on 20 Nov: 5 changed, 1 dropped); auto_class keeps contradictions.py's label.
"""

import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from contradictions import AMBIGUITY, call
from dotenv import dotenv_values
from field_match import HERE, ROOT
from google import genai
from google.genai import errors as genai_errors
from llm_cache import JsonCache
from pydantic import BaseModel
from sqlmodel import Session

from train_rag.database import get_engine
from train_rag.evaluation.usage import UsageMeter
from train_rag.processing.normalize import normalize_content
from train_rag.repository.document import ids_of_doc_ids, load_documents

DATASET = HERE / "data" / "dev_contradictions.jsonl"
CACHE = HERE / "data" / "agreement_cache.json"
SEED = 31
# How much of each document the classifier sees: the whole text (capped), the first 3,000
# characters (the baseline run) or the agent's snippet (its first 300 characters).
MODES = {"full": 12_000, "head": 3_000, "snippet": 300}
WORKERS = 8
CLASS_OF = {
    "same_answer": "consistent", "one_case_fits": "specific", "several_cases_fit": "ambiguous",
}
CLASSES = ["consistent", "specific", "ambiguous"]

PROMPTS = {
    "baseline": """You check search results before anyone answers a question from them. You
get a question and the internal company documents search returned (chat threads, emails,
tickets, docs). Some may not be about the question at all.

Decide whether the answer depends on which document is picked:
- same_answer: the documents that answer the question say the same thing (or only one
  document answers it);
- one_case_fits: the documents describe different cases (another incident, customer,
  release or decision), but the question's details single out one of them;
- several_cases_fit: different cases with different answers fit the question as asked, so
  the person must be asked which one they mean.

In reasoning, say briefly which documents answer the question and what case each is about.
Then give the verdict and, in fits, the ids of the documents that fit the question as
asked. Do not answer the question.
""",
    "v2": """You check search results before anyone answers a question from them. You
get a question and the internal company documents search returned (chat threads, emails,
tickets, docs). Many are on the same topic; some are not about the question at all.

Your job: would the person get a different answer depending on which document is used?
Do not answer the question.

Steps:
1. Find the documents that answer the question as asked (fully or mostly). Ignore the
   rest.
2. Group them by case: one case is one incident, customer request, release, design
   discussion or decision. Several threads about the same case (a follow-up, a later
   iteration of the same design) are one case if what they say is compatible.
3. For each group, note the facts the question asks for (the fix, cause, plan, owner,
   ticket, date, number). Two groups differ only if those facts differ in substance, not
   in wording or level of detail.
4. Check the question's own details (customer, team, channel, environment, region,
   symptom, time, project name). A detail rules a case out only if that case clearly
   doesn't match it. If a case also matches the detail, or doesn't say, it still fits.
   A shared channel or topic doesn't tell two threads apart.

Verdict:
- same_answer: one case answers, and no other case on the same topic comes close; or
  several documents answer and their answers agree in substance.
- one_case_fits: other documents describe different cases on the same topic, with
  different answers, but the question's details rule all of them out but one.
- several_cases_fit: two or more cases with different answers still fit the question
  after step 4. The person has to be asked which one they mean.

In reasoning, list the answering documents by case, what differs between the cases, and
which detail of the question (if any) rules cases out. In fits, give the ids of the
documents that fit the question as asked.
""",
    "v3": """You check search results before anyone answers a question from them. You
get a question and the internal company documents search returned (chat threads, emails,
tickets, docs). Some may not be about the question at all.

First go through every document, in order. For each one give:
- fits: could the person asking mean this document? "yes" if it is about what the question
  asks, even if it doesn't answer every part of it; "partly" if it touches the question; "no"
  if it is about something else.
- case: in a few words, which case it is about (which incident, customer, release, feature or
  decision, and when, if it says).
- answer: in one line, what it says to the question (the fix, cause, plan, owner, number).
  Empty if fits is "no".

Then decide whether the answer depends on which document is picked, looking only at the
documents that fit:
- same_answer: they say the same thing in substance (wording, detail and later iterations
  of the same discussion don't matter), or only one document fits;
- one_case_fits: they describe different cases with different answers, but the question's
  details single out one of them;
- several_cases_fit: different cases with different answers fit the question as asked, so
  the person must be asked which one they mean. A detail in the question singles out a case
  only if the other cases clearly don't match it.

In reasoning, say which cases fit and what separates them. Give the verdict and, in fits,
the ids of the documents that fit the question as asked. Do not answer the question.
""",
    "v4": """You check search results before anyone answers a question from them. You
get a question and the internal company documents search returned (chat threads, emails,
tickets, docs). Some may not be about the question at all.

First go through every document, in order, and judge each one on its own, as if it were
the only result: don't compare it with the other documents yet. For each one give:
- fits: if this were the only document, would it answer the question? "yes" if it answers
  what the question asks (a person could have meant this case), even if it misses a detail
  or doesn't answer every part; "partly" if it touches the question; "no" if it is about
  something else or contradicts a detail the question states.
- case: in a few words, which case it is about (which incident, customer, release, feature or
  decision, and when, if it says).
- answer: in one line, what it says to the question (the fix, cause, plan, owner, number).
  Empty if fits is "no".

Then compare the documents that fit (yes):
- same_answer: they are about the same case and agree in substance (wording, detail and later
  iterations of the same discussion don't matter), or only one document fits;
- one_case_fits: they describe different cases with different answers, but a detail the
  question states rules out all of them but one (the others clearly contradict it, not just
  leave it unsaid);
- several_cases_fit: two or more different cases with different answers fit. One matching
  the question's wording better is not enough to pick it: the person must be asked which
  case they mean.

In reasoning, say which cases fit and what separates them. Give the verdict and, in fits,
the ids of the documents that fit the question as asked. Do not answer the question.
""",
}


class Verdict(BaseModel):
    """Whether the answer depends on which document is picked."""

    reasoning: str
    verdict: Literal["same_answer", "one_case_fits", "several_cases_fit"]
    fits: list[str]


class DocumentCase(BaseModel):
    """One document as the v3 prompt reads it, before the verdict."""

    id: str
    fits: Literal["yes", "partly", "no"]
    case: str
    answer: str


class CaseVerdict(BaseModel):
    """The v3 output: every document's case and answer first, then the verdict.

    Field order is the order Gemini writes them in, so it reads the documents before it
    decides.
    """

    documents: list[DocumentCase]
    reasoning: str
    verdict: Literal["same_answer", "one_case_fits", "several_cases_fit"]
    fits: list[str]


# The output schema per prompt version; Verdict unless listed.
SCHEMAS: dict[str, type[Verdict | CaseVerdict]] = {"v3": CaseVerdict, "v4": CaseVerdict}


def half_f1(rows: list[dict[str, Any]]) -> float:
    """Ask F1 (ambiguous vs the rest) on some scored questions."""
    tp = sum(r["class"] == r["predicted"] == "ambiguous" for r in rows)
    fp = sum(r["class"] != "ambiguous" and r["predicted"] == "ambiguous" for r in rows)
    fn = sum(r["class"] == "ambiguous" and r["predicted"] != "ambiguous" for r in rows)
    return round(2 * tp / max(2 * tp + fp + fn, 1), 3)


def main(version: str, mode: str, sample: int = 0) -> None:
    """Classify every question of the set and score the verdicts."""
    chars = MODES[mode]

    def body(content: str) -> str:
        text = normalize_content(content)
        if mode == "snippet":  # as the agent sees it: whitespace collapsed
            return " ".join(text.split())[:chars] + "…"
        return text[:chars]

    env = dotenv_values(ROOT / ".env")
    rows = [json.loads(line) for line in DATASET.open()]
    ambiguity = json.loads(AMBIGUITY.read_text())["questions"]
    shown = {}
    for r in rows:
        seen = ambiguity[r["right_doc_id"]]
        shown[r["id"]] = sorted({d for d, _ in seen["today"] + seen["agent"]} | {r["right_doc_id"]})
    with Session(get_engine()) as session:
        wanted = sorted({d for ids in shown.values() for d in ids})
        documents = {
            doc.doc_id: doc
            for doc in load_documents(
                session, list(ids_of_doc_ids(session, "enterprise_rag_bench", wanted))
            )
        }

    client = genai.Client(api_key=env["GEMINI_API_KEY"])
    usage = UsageMeter()
    # Repeat k > 0: a fresh cache, so Gemini answers again (run-to-run noise).
    cache = JsonCache(CACHE.with_stem(f"{CACHE.stem}_s{sample}") if sample else CACHE)

    Result = tuple[dict[str, Any], dict[str, str], Verdict | CaseVerdict | None]

    def classify(row: dict[str, Any]) -> Result:
        ids = [d for d in shown[row["id"]] if d in documents]
        random.Random(f"{SEED}:{row['id']}").shuffle(ids)
        label = {d: f"D{i + 1}" for i, d in enumerate(ids)}
        text = f"Question: {row['question']}\n\n" + "\n\n".join(
            f"[{label[d]}] {documents[d].source_type}: {documents[d].title}\n"
            f"{body(documents[d].content)}"
            for d in ids
        )
        try:
            schema = SCHEMAS.get(version, Verdict)
            verdict = call(client, usage, cache, PROMPTS[version], text, schema)
        except genai_errors.APIError:
            return row, label, None
        assert isinstance(verdict, Verdict | CaseVerdict)
        return row, label, verdict

    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(classify, rows))
    cache.save()

    confusion: Counter[tuple[str, str]] = Counter()
    found: Counter[str] = Counter()
    clean: Counter[str] = Counter()
    out = []
    for row, label, verdict in results:
        if verdict is None:
            continue
        doc_of = {v: k for k, v in label.items()}
        fits = {doc_of[f] for f in verdict.fits if f in doc_of}
        predicted = CLASS_OF[verdict.verdict]
        confusion[(row["class"], predicted)] += 1
        found[row["class"]] += row["right_doc_id"] in fits
        conflicting = {c["doc_id"] for c in row["conflicting"]}
        clean[row["class"]] += row["right_doc_id"] in fits and not fits & conflicting
        out.append({
            "id": row["id"], "class": row["class"], "predicted": predicted,
            "question": row["question"], "reasoning": verdict.reasoning,
            "fits": sorted(fits), "right_doc_id": row["right_doc_id"],
            "documents": [d.model_dump() for d in getattr(verdict, "documents", [])],
            "labels": label,
        })

    total = sum(confusion.values())
    correct = sum(confusion[(c, c)] for c in CLASSES)
    ask_tp = confusion[("ambiguous", "ambiguous")]
    ask_fp = sum(confusion[(c, "ambiguous")] for c in CLASSES if c != "ambiguous")
    ask_fn = sum(confusion[("ambiguous", c)] for c in CLASSES if c != "ambiguous")
    recall = {
        c: confusion[(c, c)] / max(sum(confusion[(c, p)] for p in CLASSES), 1) for c in CLASSES
    }
    report = {
        "version": version, "sample": sample,
        "mode": mode,
        "per-class recall": {c: round(v, 3) for c, v in recall.items()},
        "macro recall": round(sum(recall.values()) / len(CLASSES), 3),
        "answered": total,
        "accuracy": round(correct / total, 3),
        "confusion (true -> predicted)": {
            c: {p: confusion[(c, p)] for p in CLASSES} for c in CLASSES
        },
        "ask precision": round(ask_tp / max(ask_tp + ask_fp, 1), 3),
        "ask recall": round(ask_tp / max(ask_tp + ask_fn, 1), 3),
        "ask F1": round(2 * ask_tp / max(2 * ask_tp + ask_fp + ask_fn, 1), 3),
        # Prompts are tuned on this set: the F1 on each half (alternate questions) shows
        # whether a change helps both halves or only some questions.
        "ask F1 by half": [half_f1(out[k::2]) for k in (0, 1)],
        "right thread among fits": dict(found),
        "right thread fits, no conflicting one": dict(clean),
        "gemini calls": usage.calls,
        "cost ($)": round(usage.cost(), 3),
    }
    suffix = f"_s{sample}" if sample else ""
    (HERE / "data" / f"agreement_{version}_{mode}{suffix}_result.json").write_text(
        json.dumps({"report": report, "questions": out}, indent=1, ensure_ascii=False)
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:] + ["baseline", "full", "0"][len(sys.argv) - 1 :]
    main(args[0], args[1], int(args[2]))
