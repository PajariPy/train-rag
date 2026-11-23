"""Export every gte chunk of the corpus, so a rented GPU can embed it with the new model.

Writes data/corpus.jsonl.gz: one line per chunk with its embedding table, row id and text,
the exact text production embedded with gte-modernbert-base. The vectors that come back
(embed_corpus.py) are stored as new rows that copy everything else from these rows.

Free, local, a few minutes; about 2.5 GB of text, ~0.8 GB compressed.

Usage:
    uv run python exploration/embedder_finetune/export_corpus.py
"""

import gzip
import json
from pathlib import Path

from sqlalchemy import text

from train_rag.database import get_engine
from train_rag.tables import EMBEDDING_TABLES

DATA = Path(__file__).parent / "data"
OUTPUT = DATA / "corpus.jsonl.gz"
MODEL = "gte-modernbert-base"
BATCH = 20_000


def main() -> None:
    """Stream every chunk of every 768-dimension table into one compressed file."""
    DATA.mkdir(exist_ok=True)
    written = 0

    with get_engine().connect() as connection, gzip.open(OUTPUT, "wt") as output:
        for table in EMBEDDING_TABLES[768].values():
            name = table.__tablename__
            rows = connection.execution_options(stream_results=True).execute(
                text(
                    f"select id, text from {name} where model = :model and field = 'chunk' "
                    "order by id"
                ),
                {"model": MODEL},
            )

            for batch in rows.partitions(BATCH):
                for row in batch:
                    line = {"table": name, "id": row.id, "text": row.text}
                    output.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += len(batch)

            print(f"{name}: done, {written:,} chunks so far", flush=True)

    print(f"{written:,} chunks written to {OUTPUT}")


if __name__ == "__main__":
    main()
