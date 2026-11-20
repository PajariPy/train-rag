"""Add ~2,000 training documents from the sources that are thin in the SLM's training set.

Gmail is 24% of the corpus but 9% of the training labels; HubSpot, GitHub, Jira,
Confluence and Fireflies have fewer than 500 each. This picks random documents from those
sources (fixed seed), away from every evaluation set and from everything already in
data/inputs.jsonl, and appends them to inputs.jsonl with use `train` and batch
`extra-2026-11-20`. Then label.py --use train labels only the new ones (labelled keys are
skipped).

Free and local (database only). Usage:
    uv run python exploration/slack_extraction/add_training_docs.py
"""

import json
import random
from collections import Counter

from build_inputs import DATASET, EXCLUDED, OUTPUT, document_text
from sqlmodel import Session, select

from train_rag.database import get_engine
from train_rag.models import SourceType
from train_rag.parsers import default_registry
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.repository.inferred_messages import load_inferred_messages
from train_rag.tables import DocumentRecord

SEED = 20
BATCH = "extra-2026-11-20"
PER_SOURCE = {
    SourceType.GMAIL: 1_000,
    SourceType.HUBSPOT: 400,
    SourceType.GITHUB: 200,
    SourceType.JIRA: 150,
    SourceType.CONFLUENCE: 150,
    SourceType.FIREFLIES: 100,
}


def main() -> None:
    """Pick the documents, load their texts and append them to inputs.jsonl."""
    rng = random.Random(SEED)
    with OUTPUT.open() as lines:
        existing = [json.loads(line) for line in lines]
    if any(item.get("batch") == BATCH for item in existing):
        raise SystemExit(f"batch {BATCH} is already in {OUTPUT}")
    blocked = {item["id"] for item in existing} | set(
        json.loads(EXCLUDED.read_text())["evaluation"]
    )

    picked: list[str] = []
    with Session(get_engine()) as session:
        for source, count in PER_SOURCE.items():
            ids = session.exec(
                select(DocumentRecord.doc_id).where(
                    DocumentRecord.dataset == DATASET, DocumentRecord.source_type == source
                )
            ).all()
            pool = sorted(set(ids) - blocked)
            picked += rng.sample(pool, count)
        registry = default_registry(inferred_messages=load_inferred_messages(session))
        documents = load_documents(session, list(ids_of_doc_ids(session, DATASET, picked)))

    unique = {document.doc_id: document for document in documents}
    items = [
        {"id": doc.doc_id, "use": "train", "source": str(doc.source_type),
         "batch": BATCH, "text": document_text(doc, registry)}
        for doc in unique.values()
    ]
    with OUTPUT.open("a") as file:
        file.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items)

    print(f"appended {len(items):,} items to {OUTPUT}:", dict(Counter(i["source"] for i in items)))


if __name__ == "__main__":
    main()
