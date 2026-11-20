"""Agent v3 with clarifying questions, on the 200 Slack dev questions.

The evaluation with clarifying questions (docs/superpowers/specs/2026-11-09-clarification-
eval-design.md) moved to the field-search agent and the Slack dev set:

1. A memory note per question: what the person who asked remembers about the thread (kind of
   document, team, roughly when, what it was about; no names, ids or numbers). A quarter of
   the notes (fixed seed) hold one slightly wrong detail. Written by Gemini with the existing
   prompts (memory_note_*), from the question and the right thread; the dev set has no gold
   answers, so the writer is told only to avoid the thread's specifics. Cached in
   data/memory_notes.json.
2. The simulated person (evaluation/person.py, Gemini) answers only from that note.
3. Agent v3 (agent_search.py) gets `ask_user`, at most MAX_ASKS times, on top of its 4
   searches. Everything else is the same as the v3 run.

Reported like agent_search.py, next to v3 without asking, split by today's reached / missed
and by clean / wrong notes.

Paid: Gemini (notes, replies and, by default, the agent; --model deepseek for the earlier
agent). Usage:
    uv run python exploration/slack_extraction/clarify_agent.py [--limit N] [--model M]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from agent_search import (
    AGENT_ERRORS,
    INPUTS,
    QUESTIONS,
    WORKERS,
    FieldIndex,
    make_chat,
    run_agent,
)
from dotenv import dotenv_values
from field_match import BASELINE, HERE
from google import genai
from google.genai import types
from label import ENV
from pydantic import BaseModel

from train_rag.evaluation.memory_notes import pick_wrong
from train_rag.evaluation.person import SimulatedPerson
from train_rag.evaluation.pipeline_answer import GEMINI_MODEL
from train_rag.evaluation.usage import UsageMeter
from train_rag.models.memory import MemoryNote
from train_rag.prompts import prompt

NOTES = HERE / "data" / "memory_notes.json"
# Compared with the same model's v4 run without asking (agent_search.py).
V3 = {
    "deepseek": HERE / "data" / "agent_search_v4_result.json",
    "gemini": HERE / "data" / "agent_search_v4_gemini_result.json",
}
RESULT = {
    "deepseek": HERE / "data" / "clarify_agent_v4_result.json",
    "gemini": HERE / "data" / "clarify_agent_v4_gemini_result.json",
}
MAX_ASKS = 2
NO_GOLD = "(not given: keep to the big picture and leave out every specific of the document)"


class NoteDraft(BaseModel):
    """The note writer's reply."""

    note: str
    wrong_detail: str | None = None


def write_notes(
    client: genai.Client, usage: UsageMeter, questions: dict[str, str], texts: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Write (or load) the memory note of every question."""
    notes: dict[str, dict[str, Any]] = json.loads(NOTES.read_text()) if NOTES.exists() else {}
    wrong = pick_wrong(sorted(questions))
    todo = [q for q in questions if q not in notes]

    def one(doc_id: str) -> tuple[str, dict[str, Any]]:
        is_wrong = doc_id in wrong
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt(
                "memory_note_turn",
                question=questions[doc_id],
                gold_answer=NO_GOLD,
                instruction=prompt("memory_note_wrong" if is_wrong else "memory_note_clean"),
                document=texts[doc_id],
            ),
            config=types.GenerateContentConfig(
                system_instruction=prompt("memory_note_instructions"),
                response_mime_type="application/json",
                response_schema=NoteDraft,
            ),
        )
        usage.add(response)
        draft = NoteDraft.model_validate_json(response.text or "")

        return doc_id, {
            "note": draft.note.strip(),
            "wrong_detail": draft.wrong_detail if is_wrong else None,
        }

    if todo:
        print(f"writing {len(todo)} memory notes", flush=True)
        with ThreadPoolExecutor(8) as pool:
            notes |= dict(pool.map(one, todo))
        NOTES.write_text(json.dumps(notes, indent=1, ensure_ascii=False))

    return notes


def main() -> None:
    """Write the notes, run the asking agent and compare it with v3."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", choices=("gemini", "deepseek"), default="gemini")
    args = parser.parse_args()

    env = dotenv_values(ENV)
    baseline: dict[str, dict[str, Any]] = json.loads(BASELINE.read_text())
    rows = [json.loads(line) for line in QUESTIONS.open()]
    questions = {row["doc_id"]: row["question"] for row in rows if row["split"] == "dev_slack"}
    texts: dict[str, str] = {}
    with INPUTS.open() as file:
        for line in file:
            row = json.loads(line)
            if row["id"] in questions:
                texts[row["id"]] = row["text"]

    gemini = genai.Client(api_key=env["GEMINI_API_KEY"])
    gemini_usage = UsageMeter()
    notes = write_notes(gemini, gemini_usage, questions, texts)

    index = FieldIndex()
    position = {item: i for i, item in enumerate(index.pool)}
    earlier = json.loads(V3[args.model].read_text())["runs"]
    v3 = {r["doc_id"]: r for r in earlier if "error" not in r}
    todo = [q for q in sorted(v3) if q in position]
    todo = todo[: args.limit] if args.limit else todo
    chat = make_chat(args.model)

    def one(doc_id: str) -> dict[str, Any]:
        note = MemoryNote(question_id=doc_id, **notes[doc_id])
        person = SimulatedPerson(gemini, gemini_usage, questions[doc_id], note)
        try:
            ranked, searches, usage, picked, asks = run_agent(
                chat, index, questions[doc_id], person, MAX_ASKS
            )
        except AGENT_ERRORS as error:
            return {"doc_id": doc_id, "error": str(error)[:300]}

        target = position[doc_id]
        return {
            "doc_id": doc_id, "rank": ranked.index(target) + 1 if target in ranked else None,
            "picked_right": target in picked, "picked": [index.pool[i] for i in picked],
            "searches": searches, "asks": asks, "usage": usage,
            "group": "reached" if doc_id in baseline[doc_id]["scores"] else "missed",
            "wrong_note": notes[doc_id]["wrong_detail"] is not None,
        }

    with ThreadPoolExecutor(WORKERS) as pool:
        runs = list(pool.map(one, todo))

    done = [r for r in runs if "error" not in r]
    agent_cost = sum(r["usage"]["cost"] for r in done)

    def score(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        ranks = [r["rank"] for r in group]
        picked = round(100 * np.mean([r["picked_right"] for r in group]), 1)
        return {"n": len(group), "picked": picked} | {
            f"top{d}": round(100 * sum(1 for x in ranks if x and x <= d) / len(group), 1)
            for d in (10, 50, 100)
        }

    report: dict[str, Any] = {
        "questions": len(done), "errors": len(runs) - len(done),
        "model": args.model, "agent cost": round(agent_cost, 3),
        "person and notes cost (gemini)": round(gemini_usage.cost(), 3),
        "asked at least once": round(100 * np.mean([bool(r["asks"]) for r in done]), 1),
        "asks per question": round(float(np.mean([len(r["asks"]) for r in done])), 2),
    }
    asked = {r["doc_id"] for r in done}
    for group in ("reached", "missed"):
        report[f"{group}: v3"] = score([v3[q] for q in asked if v3[q]["group"] == group])
        report[f"{group}: v3 + ask"] = score([r for r in done if r["group"] == group])
    for wrong in (False, True):
        name = "wrong notes" if wrong else "clean notes"
        mine = [r for r in done if r["wrong_note"] == wrong]
        report[f"{name}: v3"] = score([v3[r["doc_id"]] for r in mine])
        report[f"{name}: v3 + ask"] = score(mine)

    RESULT[args.model].write_text(
        json.dumps({"report": report, "runs": runs}, indent=1, ensure_ascii=False)
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
