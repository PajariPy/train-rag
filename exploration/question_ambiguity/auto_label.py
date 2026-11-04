"""Label the sampled questions for ambiguity from what each shown document answers.

Rules, from what Gemini says each document answers on its own (doc_answers.json):

- A question is **ambiguous** when a document of gte's top 2 that is not the expected one
  answers it fully or partly (a close version, or part of the answer, is enough for a
  plausible wrong answer); otherwise it is **not ambiguous**.
- Flags add detail:
  - `both_answered`: neither top 2 document is the expected one, and both answer;
  - `partial`: the other documents that answer do so only partly;
  - `low_similarity`: an answering document's gte similarity is below LOW_SIMILARITY, so the
    question is ambiguous, but by a weaker match;
  - `problematic`: Gemini can't answer the question even from the expected document, so the
    question or its gold answer may be faulty;
  - `expected_partial`: the expected document answers it only partly.

The labels are written to auto_labels.json and compared with the hand labels in
labels.json.

Usage: uv run python exploration/question_ambiguity/auto_label.py
"""

import json
from pathlib import Path

from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.repository.question import load_questions

HERE = Path(__file__).parent
SAMPLE = HERE / "sample.json"
ANSWERS = HERE / "doc_answers.json"
LABELS = HERE / "labels.json"
OUT = HERE / "auto_labels.json"
# Between the sample's low group (0.675-0.706) and its middle group (about 0.77).
LOW_SIMILARITY = 0.75


def auto_label(
    expected: list[str], top2: list[dict[str, object]], answers: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Label one question from its expected documents, gte's top 2 and the answers.

    Args:
        expected: The expected documents' ids.
        top2: gte's top 2 documents, each with its `doc_id` and `gte_similarity`.
        answers: Gemini's answer from each shown document, keyed by document id.

    Returns:
        The label, its flags, and the other documents that answer.
    """
    others = [hit for hit in top2 if hit["doc_id"] not in expected]
    answering = [hit for hit in others if answers[str(hit["doc_id"])]["coverage"] != "none"]
    flags = []
    if len(others) == 2 and len(answering) == 2:
        flags.append("both_answered")
    if answering and all(answers[str(hit["doc_id"])]["coverage"] == "partial" for hit in answering):
        flags.append("partial")
    if any(float(str(hit["gte_similarity"])) < LOW_SIMILARITY for hit in answering):
        flags.append("low_similarity")
    coverage = {answers[doc_id]["coverage"] for doc_id in expected if doc_id in answers}
    if coverage <= {"none"}:
        flags.append("problematic")
    elif "full" not in coverage:
        flags.append("expected_partial")

    return {
        "label": "ambiguous" if answering else "not ambiguous",
        "flags": flags,
        "answering": [str(hit["doc_id"]) for hit in answering],
    }


def main() -> None:
    """Label every sampled question and compare with the hand labels."""
    sample = json.loads(SAMPLE.read_text())
    answers = json.loads(ANSWERS.read_text())
    with Session(get_engine()) as session:
        questions = {
            question.id: question
            for question in load_questions(session, EnterpriseRagBenchLoader.name)
        }
    labels = {
        row["question_id"]: auto_label(
            questions[row["question_id"]].expected_doc_ids,
            row["top2"],
            answers[row["question_id"]],
        )
        for row in sample
    }
    OUT.write_text(json.dumps(labels, indent=1, sort_keys=True))

    groups = {row["question_id"]: row["group"] for row in sample}
    print(f"{'question':10} {'group':7} {'auto':14} flags")
    for question_id, label in sorted(labels.items(), key=lambda item: groups[item[0]]):
        print(
            f"{question_id:10} {groups[question_id]:7} {label['label']:14} "
            f"{', '.join(label['flags'])}"
        )
    hand = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    shared = [
        (question_id, hand[question_id]["label"], labels[question_id]["label"])
        for question_id in labels
        if question_id in hand and hand[question_id]["label"] != "unsure"
    ]
    agree = sum(mine == auto for _, mine, auto in shared)
    print(f"\nHand labels to compare: {len(shared)}; the rules agree on {agree}.")
    for question_id, mine, auto in shared:
        if mine != auto:
            print(f"  {question_id}: hand {mine}, rules {auto}")


if __name__ == "__main__":
    main()
