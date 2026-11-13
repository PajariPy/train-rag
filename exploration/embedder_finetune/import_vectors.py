"""Load the fine-tuned model's corpus vectors into their own `<source>_embeddings_768_ft` tables.

Reads the part_*.npz files of embed_corpus.py. Each vector belongs to one row of the base
model's table (same table, same row id); the new row copies that row's document, position,
neighbours and text, and gets the new model's name and vector. Row ids are kept, so each
chunk's links to its neighbours (`previous_id`, `next_id`) stay right.

Steps: create the tables, drop their vector index, load every part (resumable: rows already
there are skipped), rebuild the StreamingDiskANN index.

Free, local; tens of minutes, mostly the index build.

Usage:
    uv run python exploration/embedder_finetune/import_vectors.py --vectors DIR
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sqlalchemy import Connection, text
from sqlmodel import Session

from train_rag.database import create_tables, get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.progress.recorder import RunProgress
from train_rag.repository.embedding import create_vector_index, drop_vector_index
from train_rag.tables import FINETUNED_TABLES

BATCH = 5_000
FINETUNED = {
    table.__tablename__.removesuffix("_ft"): table.__tablename__
    for table in FINETUNED_TABLES.values()
}


def load_part(connection: Connection, ids: np.ndarray, vectors: np.ndarray, base: str) -> None:
    """Insert one table's vectors of a part, copying everything else from the base rows.

    Args:
        connection: Open connection (inside a transaction).
        ids: Row ids in the base table.
        vectors: One unit vector per id.
        base: The base model's table name.
    """
    connection.execute(
        text(
            "create temp table if not exists staging "
            "(id bigint primary key, embedding vector(768)) "
            "on commit delete rows"
        )
    )
    connection.execute(text("truncate staging"))  # a part holds rows of several tables

    for first in range(0, len(ids), BATCH):
        connection.execute(
            text("insert into staging (id, embedding) values (:id, cast(:embedding as vector))"),
            [
                {"id": int(row_id), "embedding": str(vector.astype(np.float32).tolist())}
                for row_id, vector in zip(
                    ids[first : first + BATCH], vectors[first : first + BATCH], strict=True
                )
            ],
        )

    connection.execute(
        text(
            f"insert into {FINETUNED[base]} (id, document_id, doc_id, field, position, "
            "previous_id, next_id, model, text, embedding) "
            "select b.id, b.document_id, b.doc_id, b.field, b.position, b.previous_id, "
            "b.next_id, :model, b.text, s.embedding "
            f"from staging s join {base} b on b.id = s.id "
            "on conflict (id) do nothing"
        ),
        {"model": FineTunedGteEmbedder.name},
    )


def main() -> None:
    """Create the tables, load every part, build the indexes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, required=True)
    args = parser.parse_args()
    parts = sorted(args.vectors.glob("part_*.npz"))
    engine = get_engine()

    # 1. Tables, without their vector index while loading (it would update row by row).
    create_tables(engine)

    with Session(engine) as session:
        for table in FINETUNED_TABLES.values():
            drop_vector_index(session, table)

    # 2. Load every part; each in its own transaction, so a rerun resumes.
    with RunProgress(
        engine, "load fine-tuned corpus vectors (free)", len(parts), unit="parts"
    ) as progress:
        for path in parts:
            part = np.load(path)
            rows: dict[str, list[int]] = defaultdict(list)

            for index, table in enumerate(part["tables"]):
                rows[str(table)].append(index)

            with engine.begin() as connection:
                for base, indexes in rows.items():
                    load_part(connection, part["ids"][indexes], part["vectors"][indexes], base)

            progress.advance("load", 1, note=path.name)

    # 3. Indexes, one table at a time.
    with Session(engine) as session:
        for table in FINETUNED_TABLES.values():
            print(f"indexing {table.__tablename__}", flush=True)
            create_vector_index(session, table)

    with engine.connect() as connection:
        for name in FINETUNED.values():
            count = connection.execute(text(f"select count(*) from {name}")).scalar()
            print(f"{name}: {count:,} rows")


if __name__ == "__main__":
    main()
