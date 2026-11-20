"""Collect everything the field test labels, with what each item may be used for.

One list, data/inputs.jsonl, for label.py: every item has an id, its text and a `use`:

- `test`: candidates and right threads of half B of the Slack dev questions, and the 200 dev
  questions; used by the field test and the later search checks, never for training.
- `train`: candidates of half A (minus anything in half B's lists), 1,000 random Slack threads
  from outside every evaluation set, and 1,000 train-split questions; SLM training data.
- `extraction_test`: the 200 threads in data/threads.json; the SLM's extraction score.

Half A / half B is a fixed, seeded split of the 200 dev questions. A right thread (the thread a
question was written from) is always `test`, also in half A. Questions get the id
`question:<doc_id>` (dev) or `question:train:<n>` and the text "Search question: ...".

Free and local (database only). Usage:
    uv run python exploration/slack_extraction/build_inputs.py
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from sample import thread_text
from sqlmodel import Session, select

from train_rag.database import get_engine
from train_rag.models import SourceType
from train_rag.parsers import ParserRegistry, default_registry
from train_rag.processing.normalize import normalize_content
from train_rag.repository.document import ids_of_doc_ids, load_documents, to_artifact
from train_rag.repository.inferred_messages import load_inferred_messages
from train_rag.tables import DocumentRecord

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
DATASET = "enterprise_rag_bench"
BASELINE = ROOT / "exploration" / "query_expansion" / "data" / "slack_dev_baseline.json"
QUESTIONS = ROOT / "exploration" / "training_questions" / "data" / "questions.jsonl"
EXCLUDED = ROOT / "exploration" / "training_questions" / "data" / "excluded.json"
THREADS = HERE / "data" / "threads.json"
OUTPUT = HERE / "data" / "inputs.jsonl"
SPLIT = HERE / "data" / "question_halves.json"
SEED = 17
RANDOM_THREADS = 1_000
TRAIN_QUESTIONS = 1_000


def document_text(document: DocumentRecord, registry: ParserRegistry) -> str:
    """A document as DeepSeek reads it: Slack as message lines, the rest as title + text.

    Args:
        document: The stored document.
        registry: The parsers, for Slack threads.

    Returns:
        The text to label.
    """
    if document.source_type == SourceType.SLACK:
        text, channel, messages = thread_text(registry, to_artifact(document))

        if messages:
            return f"Channel: #{channel}\n{text}" if channel else text

    content = normalize_content(document.content)

    return f"{document.title}\n\n{content}" if document.title else content


def main() -> None:
    """Pick the items, load their texts and write data/inputs.jsonl."""
    rng = random.Random(SEED)
    baseline: dict[str, dict[str, Any]] = json.loads(BASELINE.read_text())
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines()]
    dev = {row["doc_id"]: row["question"] for row in rows if row["split"] == "dev_slack"}

    # Half A trains, half B is held out; a document in any half-B list stays held out.
    order = sorted(dev)
    rng.shuffle(order)
    half_a, half_b = set(order[:100]), set(order[100:])
    SPLIT.write_text(json.dumps({"A": sorted(half_a), "B": sorted(half_b)}, indent=1))

    held_out = set(dev)
    for doc_id in half_b:
        held_out |= set(baseline[doc_id]["scores"])

    candidates_a = set()
    for doc_id in half_a:
        candidates_a |= set(baseline[doc_id]["scores"])

    uses: dict[str, str] = {doc_id: "test" for doc_id in held_out}
    uses |= {doc_id: "train" for doc_id in candidates_a - held_out}

    extraction = {thread["doc_id"] for thread in json.loads(THREADS.read_text())}
    uses |= {doc_id: "extraction_test" for doc_id in extraction}

    # Random Slack threads, away from every evaluation set and everything already picked.
    blocked = set(json.loads(EXCLUDED.read_text())["evaluation"]) | set(uses)

    with Session(get_engine()) as session:
        slack_ids = session.exec(
            select(DocumentRecord.doc_id).where(
                DocumentRecord.dataset == DATASET,
                DocumentRecord.source_type == SourceType.SLACK,
            )
        ).all()
        pool = sorted(set(slack_ids) - blocked)
        uses |= {doc_id: "train" for doc_id in rng.sample(pool, RANDOM_THREADS)}

        print(f"loading {len(uses):,} documents", flush=True)
        registry = default_registry(inferred_messages=load_inferred_messages(session))
        documents = load_documents(session, list(ids_of_doc_ids(session, DATASET, list(uses))))

    unique = {document.doc_id: document for document in documents}  # one doc_id is stored twice
    items = [
        {"id": document.doc_id, "use": uses[document.doc_id],
         "source": str(document.source_type), "text": document_text(document, registry)}
        for document in unique.values()
    ]

    # Questions: all 200 dev questions (test) and a sample of train-split questions.
    items += [
        {"id": f"question:{doc_id}", "use": "test", "source": "question",
         "text": f"Search question: {question}"}
        for doc_id, question in sorted(dev.items())
    ]
    train_questions = [row["question"] for row in rows if row["split"] == "train"]
    items += [
        {"id": f"question:train:{number}", "use": "train", "source": "question",
         "text": f"Search question: {question}"}
        for number, question in enumerate(rng.sample(train_questions, TRAIN_QUESTIONS))
    ]

    OUTPUT.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))

    counts = Counter((item["use"], item["source"] == "question") for item in items)
    missing = len(uses) - len(unique)
    print(f"wrote {len(items):,} items to {OUTPUT} ({missing} documents not found)")
    for (use, is_question), count in sorted(counts.items()):
        print(f"  {use:16} {'questions' if is_question else 'documents':10} {count:,}")


if __name__ == "__main__":
    main()
