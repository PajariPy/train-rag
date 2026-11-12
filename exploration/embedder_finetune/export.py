"""Export what the embedder fine-tune needs, so training runs without the database.

Design: docs/brainstorm/005_synthetic_training_questions.md (data) and 004 (negatives).

Copies the training and dev sets from exploration/training_questions/data/ and writes the
text of every document they mention to data/documents.jsonl: positives, negatives and every
dev candidate. The text is the document's first gte chunk exactly as production embedded it
(title + text, up to 24,000 characters), so training sees what search sees.

Free, local, a minute or two. The output folder can be copied to a rented GPU as is.

Usage:
    uv run python exploration/embedder_finetune/export.py
"""

import json
import shutil
from pathlib import Path

from sqlalchemy import text

from train_rag.database import get_engine
from train_rag.tables import EMBEDDING_TABLES

SETS = Path(__file__).parent.parent / "training_questions" / "data"
DATA = Path(__file__).parent / "data"
FILES = ("training_set.jsonl", "dev_set.jsonl")
MODEL = "gte-modernbert-base"


def needed_documents() -> set[str]:
    """Every doc id the training and dev sets mention.

    Returns:
        Positives and negatives of both sets, and every scored dev candidate.
    """
    doc_ids: set[str] = set()

    for name in FILES:
        for line in (SETS / name).read_text().splitlines():
            row = json.loads(line)
            doc_ids.add(row["positive"])
            doc_ids.update(row["negatives"])
            if name == "dev_set.jsonl":
                doc_ids.update(row["scores"])

    return doc_ids


def first_chunks(doc_ids: set[str]) -> dict[str, str]:
    """The first gte chunk of each document, as production embedded it.

    Args:
        doc_ids: The documents.

    Returns:
        doc_id -> chunk text (title + text).
    """
    texts: dict[str, str] = {}

    with get_engine().connect() as connection:
        for table in EMBEDDING_TABLES[768].values():
            rows = connection.execute(
                text(
                    f"select doc_id, text from {table.__tablename__} "
                    "where doc_id = any(:ids) and model = :model and field = 'chunk' "
                    "and position = 0"
                ),
                {"ids": sorted(doc_ids), "model": MODEL},
            ).all()
            texts.update({str(row.doc_id): str(row.text) for row in rows})

    return texts


def main() -> None:
    """Copy the sets and write the document texts."""
    # 1. Copy the training and dev sets.
    DATA.mkdir(exist_ok=True)

    for name in FILES:
        shutil.copy(SETS / name, DATA / name)

    # 2. Look up the text of every document they mention.
    doc_ids = needed_documents()
    texts = first_chunks(doc_ids)
    missing = doc_ids - texts.keys()

    # 3. Write the texts and report what's missing.
    with (DATA / "documents.jsonl").open("w") as output:
        for doc_id, chunk in sorted(texts.items()):
            output.write(json.dumps({"doc_id": doc_id, "text": chunk}, ensure_ascii=False) + "\n")

    print(f"{len(texts):,} documents written, {len(missing)} without a gte chunk")
    if missing:
        print("missing:", sorted(missing)[:10])


if __name__ == "__main__":
    main()
