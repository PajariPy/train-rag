"""Pick the source documents for the synthetic training and dev questions.

Design: docs/brainstorm/005_synthetic_training_questions.md. Plan task 2.

1. Exclusion set: the right documents of all 500 benchmark questions and each one's 20
   nearest documents by gte (so no training question comes from a test answer or its twin).
2. Dev first: 520 documents, by question type and source in the benchmark's proportions.
   Dev documents and their 20 nearest are then excluded too.
3. Train: 2,600 documents the same way.

Miscellaneous questions come from the benchmark's off-topic folders, found through the
benchmark repo's doc_id -> path index (`data/uuid_index.json`).

Writes data/sources.json. Free, a few minutes (one vector search per excluded document).

Usage:
    uv run python exploration/training_questions/sources.py
"""

import json
import random
from pathlib import Path

from sqlalchemy import Connection, text

from train_rag.database import get_engine
from train_rag.tables import EMBEDDING_TABLES
from train_rag.training.quotas import (
    SOURCE_WEIGHTS,
    TYPE_WEIGHTS,
    is_usable_source,
    split_by_weight,
)

SEED = 7
DEV_SIZE = 520
TRAIN_SIZE = 2600
NEIGHBOURS = 20
DATA = Path(__file__).parent / "data"
UUID_INDEX = DATA / "uuid_index.json"
OUTPUT = DATA / "sources.json"
EXCLUDED = DATA / "excluded.json"
MISC_PREFIXES = (
    "slack/sports/",
    "slack/memes/",
    "confluence/misc/",
    "jira/misc-requests/",
    "fireflies/interviews/",
    "github/hackathons/",
)
MISC_PARTS = ("/misc-assets/", "/_temp_uploads/")


def nearest_documents(connection: Connection, doc_id: str) -> list[str]:
    """Find a document's nearest documents by gte, across every source.

    Args:
        connection: Open database connection.
        doc_id: The document.

    Returns:
        Up to `NEIGHBOURS` other doc ids, closest first.
    """
    tables = [table.__tablename__ for table in EMBEDDING_TABLES[768].values()]
    vector = None
    for table in tables:
        vector = connection.execute(
            text(
                f"select embedding from {table} where doc_id = :doc_id "
                "and model = 'gte-modernbert-base' and field = 'chunk' order by position limit 1"
            ),
            {"doc_id": doc_id},
        ).scalar()
        if vector is not None:
            break
    if vector is None:
        return []
    hits: list[tuple[float, str]] = []
    for table in tables:
        hits += [
            (float(row[1]), str(row[0]))
            for row in connection.execute(
                text(
                    f"select doc_id, embedding <=> cast(:vector as vector) from {table} "
                    "where model = 'gte-modernbert-base' and field = 'chunk' "
                    "order by 2 limit :limit"
                ),
                {"vector": str(vector), "limit": NEIGHBOURS + 5},
            ).all()
        ]
    ordered = [hit_id for _, hit_id in sorted(hits) if hit_id != doc_id]

    return list(dict.fromkeys(ordered))[:NEIGHBOURS]


def with_neighbours(connection: Connection, doc_ids: list[str]) -> set[str]:
    """Add each document's nearest documents to the set.

    Args:
        connection: Open database connection.
        doc_ids: The documents.

    Returns:
        The documents and all their neighbours.
    """
    result = set(doc_ids)
    for index, doc_id in enumerate(doc_ids, 1):
        result.update(nearest_documents(connection, doc_id))
        if index % 200 == 0:
            print(f"  neighbours: {index}/{len(doc_ids)}", flush=True)

    return result


def is_misc(path: str) -> bool:
    """Tell whether a benchmark path is one of the off-topic folders.

    Args:
        path: The document's path in the benchmark repo.

    Returns:
        True for miscellaneous documents.
    """
    return path.startswith(MISC_PREFIXES) or any(part in path for part in MISC_PARTS)


def sample_split(
    size: int,
    documents: dict[str, tuple[str, int]],
    misc_pool: set[str],
    excluded: set[str],
    rng: random.Random,
    misc_cap: int,
) -> list[dict[str, str]]:
    """Sample one split's sources by question type and source.

    The off-topic pool is small (89 documents in the whole corpus), so miscellaneous gets at
    most `misc_cap` and whatever is left in the pool; the shortfall goes to basic, whose
    prompt miscellaneous questions use anyway.

    Args:
        size: Number of sources.
        documents: doc_id -> (source type, character count).
        misc_pool: Off-topic documents (the only sources of miscellaneous questions).
        excluded: Documents that may not be sources.
        rng: Seeded random generator.
        misc_cap: Most miscellaneous sources to take.

    Returns:
        One row per source: doc_id, source_type, question_type.
    """
    rows: list[dict[str, str]] = []
    used = set(excluded)
    type_counts = split_by_weight(size, TYPE_WEIGHTS)
    misc_available = sorted(misc_pool - used)
    misc_count = min(type_counts["miscellaneous"], misc_cap, len(misc_available))
    type_counts["basic"] += type_counts["miscellaneous"] - misc_count
    type_counts["miscellaneous"] = misc_count
    for question_type, type_count in type_counts.items():
        if question_type == "miscellaneous":
            groups = {"misc": misc_available}
            counts = {"misc": type_count}
        else:
            counts = split_by_weight(type_count, SOURCE_WEIGHTS[question_type])
            groups = {
                source: sorted(
                    doc_id
                    for doc_id, (doc_source, chars) in documents.items()
                    if doc_source == source
                    and doc_id not in used
                    and doc_id not in misc_pool
                    and is_usable_source(chars, question_type)
                )
                for source in counts
            }
        for group, count in counts.items():
            for doc_id in rng.sample(groups[group], count):
                rows.append(
                    {
                        "doc_id": doc_id,
                        "source_type": documents[doc_id][0],
                        "question_type": question_type,
                    }
                )
                used.add(doc_id)

    return rows


def main() -> None:
    """Build the exclusion set, sample dev then train, and write the sources."""
    paths: dict[str, str] = json.loads(UUID_INDEX.read_text())
    rng = random.Random(SEED)
    with get_engine().connect() as connection:
        documents = {
            str(row.doc_id): (str(row.source_type), int(row.char_count))
            for row in connection.execute(
                text("select doc_id, source_type, char_count from documents")
            ).all()
        }
        misc_pool = {
            doc_id for doc_id, path in paths.items() if is_misc(path) and doc_id in documents
        }
        evaluation = sorted(
            {
                str(doc_id)
                for (doc_ids,) in connection.execute(
                    text("select expected_doc_ids from questions")
                ).all()
                for doc_id in doc_ids
            }
        )
        print(
            f"{len(documents):,} documents, {len(misc_pool):,} miscellaneous, "
            f"{len(evaluation)} evaluation documents"
        )
        benchmark_excluded = with_neighbours(connection, evaluation)
        misc_left = len(misc_pool - benchmark_excluded)
        dev = sample_split(
            DEV_SIZE,
            documents,
            misc_pool,
            benchmark_excluded,
            rng,
            misc_cap=misc_left * DEV_SIZE // (DEV_SIZE + TRAIN_SIZE),
        )
        dev_excluded = with_neighbours(connection, [row["doc_id"] for row in dev])
        train = sample_split(
            TRAIN_SIZE,
            documents,
            misc_pool,
            benchmark_excluded | dev_excluded,
            rng,
            misc_cap=misc_left,
        )
        print(f"miscellaneous documents left after excluding the benchmark's: {misc_left}")

    rows = [row | {"split": "dev"} for row in dev] + [row | {"split": "train"} for row in train]
    OUTPUT.write_text(json.dumps(rows, indent=1))
    EXCLUDED.write_text(json.dumps({"evaluation": evaluation}, indent=1))
    train_ids = {row["doc_id"] for row in train}
    print(f"{len(rows)} sources ({len(dev)} dev, {len(train)} train)")
    print(
        f"check, sources in the benchmark exclusion set: "
        f"{sum(row['doc_id'] in benchmark_excluded for row in rows)}"
    )
    print(f"check, train documents among dev or its neighbours: {len(train_ids & dev_excluded)}")
    print(f"check, documents used twice: {len(rows) - len({row['doc_id'] for row in rows})}")


if __name__ == "__main__":
    main()
