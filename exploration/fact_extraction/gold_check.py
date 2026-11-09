"""How much of the benchmark's gold facts does the extraction keep? A check, not a target.

For every question whose expected document has a label, each gold fact (`answer_facts`) is
looked for in the label's text (about + entities + facts). A fact counts as kept when every
number in it appears there and at least WORD_SHARE of its content words do, matched the way
BM25 matches them (lowercased, no stopwords, stemmed). The same test against the document's
full text is the ceiling: how much of the fact the document states in matching words at all.

Only reads labels; changes nothing. Usage (from the main checkout):
    uv run --project ~/projects/rag-implementation/train-rag python \
        exploration/fact_extraction/gold_check.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import Stemmer
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.index_bm25 import tokenize
from train_rag.processing.normalize import normalize_documents
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.repository.question import load_questions

HERE = Path(__file__).parent
LABELS = HERE / "data" / "labels.json"
OUT = HERE / "data" / "gold_check.json"
WORD_SHARE = 0.7
NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
STEMMER = Stemmer.Stemmer("english")


def words(text: str) -> set[str]:
    """Content words as BM25 sees them."""
    return set(tokenize([text], STEMMER)[0])


def kept(fact: str, text: str, text_words: set[str]) -> bool:
    """Whether a gold fact's numbers and most of its content words appear in a text."""
    numbers = NUMBER.findall(fact)
    if any(number not in text for number in numbers):
        return False
    wanted = words(fact)

    return not wanted or len(wanted & text_words) / len(wanted) >= WORD_SHARE


def main() -> None:
    """Compare each labelled expected document with its question's gold facts."""
    labels = {v["doc_id"]: v for v in json.loads(LABELS.read_text()).values()}
    with Session(get_engine()) as session:
        questions = [
            q
            for q in load_questions(session, EnterpriseRagBenchLoader.name)
            if q.answer_facts
            and q.expected_doc_ids
            and all(d in labels for d in q.expected_doc_ids)
        ]
        wanted = sorted({d for q in questions for d in q.expected_doc_ids})
        ids = ids_of_doc_ids(session, EnterpriseRagBenchLoader.name, wanted)
        records = load_documents(session, sorted(ids))
    full = {d.doc_id: f"{d.title}\n{d.text}" for d in normalize_documents(records)}

    rows = []
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for question in questions:
        label_text = "\n".join(
            "\n".join([labels[d]["about"], *labels[d]["entities"], *labels[d]["facts"]])
            for d in question.expected_doc_ids
        )
        doc_text = "\n".join(full.get(d, "") for d in question.expected_doc_ids)
        label_words, doc_words = words(label_text), words(doc_text)
        in_label = [kept(f, label_text, label_words) for f in question.answer_facts]
        in_doc = [kept(f, doc_text, doc_words) for f in question.answer_facts]
        kind = str(question.question_type)
        for key in (kind, "all"):
            totals[key][0] += len(question.answer_facts)
            totals[key][1] += sum(in_label)
            totals[key][2] += sum(in_doc)
        rows.append(
            {
                "question_id": question.id,
                "type": kind,
                "facts": [
                    {"fact": f, "in_label": a, "in_document": b}
                    for f, a, b in zip(question.answer_facts, in_label, in_doc, strict=True)
                ],
            }
        )

    OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"questions with every expected document labelled: {len(questions)}")
    header = ["gold facts", "kept by label", "in document", "kept / in doc"]
    print(f"{'type':26} {header[0]:>10} {header[1]:>14} {header[2]:>12} {header[3]:>14}")
    for kind, (facts, label, doc) in sorted(totals.items(), key=lambda kv: (kv[0] == "all", kv[0])):
        share = f"{label / doc:.0%}" if doc else "—"
        print(f"{kind:26} {facts:>10} {label / facts:>13.0%} {doc / facts:>11.0%} {share:>14}")
    print(f"wrote {OUT}")


main()
