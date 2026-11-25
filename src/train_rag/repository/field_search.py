"""Queries for field search's table (`field_search_vectors`): fill it, index it, search it."""

from collections import Counter
from collections.abc import Sequence

from sqlalchemy import text
from sqlmodel import Session

from train_rag.models.mined_fields import FieldDocument, MinedFields
from train_rag.repository.embedding import INDEX_MEMORY
from train_rag.tables.field_search import FieldSearchRecord

TABLE = FieldSearchRecord.__tablename__
# The indexed profile vectors, one cosine StreamingDiskANN index each. They are unit length,
# so cosine ranks like the dot product. `all` (the sum of the three cosine scores) has no
# index of its own: the compressed inner-product index ranked a summed vector almost at
# random (0.7% overlap with an exact scan, 23 Nov), so `all` merges each profile's nearest
# documents and scores them exactly.
PROFILE_VECTORS = ("subject", "problem", "names")
# Candidates read with full vectors to put the returned ones in exact order; StreamingDiskANN
# otherwise orders past its default 50 by the compressed vectors.
RESCORE = 300
# Graph candidates kept while searching (default 100); more finds more of a long top list.
SEARCH_LIST = 400


def document_id_bounds(session: Session) -> tuple[int, int]:
    """Find the lowest and highest document row id with mined fields.

    Args:
        session: Active database session.

    Returns:
        (lowest, highest), (0, -1) when nothing is mined.
    """
    low, high = session.execute(
        text("SELECT min(document_id), max(document_id) FROM document_fields")
    ).one()

    return (low, high) if low is not None else (0, -1)


def clear_field_search(session: Session, model: str) -> None:
    """Drop the search indexes and a model's rows, so a rebuild fills a bare table.

    Commits.

    Args:
        session: Active database session.
        model: Embedding model of the vectors.
    """
    for column in PROFILE_VECTORS:
        session.execute(text(f"DROP INDEX IF EXISTS {TABLE}_{column}_idx"))
    session.execute(text(f"DELETE FROM {TABLE} WHERE model = :model"), {"model": model})
    session.commit()


def fill_field_search(
    session: Session, model: str, preference: Sequence[str], low: int, high: int
) -> int:
    """Copy the preferred fields row of each document in a range, with its vectors.

    Commits.

    Args:
        session: Active database session.
        model: Embedding model of the vectors.
        preference: Extractor name prefixes, preferred first; others come last.
        low: First document row id.
        high: Document row id after the last.

    Returns:
        Rows written.
    """
    ranks = " ".join(
        f"WHEN f.extractor LIKE :prefix{place} THEN {place}" for place in range(len(preference))
    )
    statement = text(f"""
        INSERT INTO {TABLE} (document_fields_id, model, document_id, doc_id, source_type,
                             channel, areas, subject, problem, names)
        SELECT DISTINCT ON (f.document_id)
               f.id, :model, f.document_id, f.doc_id, f.source_type, coalesce(f.channel, ''),
               f.areas, v.subject, v.problem, v.names
        FROM document_fields f
        JOIN document_field_vectors v ON v.document_fields_id = f.id AND v.model = :model
        WHERE f.document_id >= :low AND f.document_id < :high
        ORDER BY f.document_id, CASE {ranks} ELSE {len(preference)} END, f.created_at DESC
    """)
    prefixes = {f"prefix{place}": f"{prefix}%" for place, prefix in enumerate(preference)}
    result = session.execute(statement, {"model": model, "low": low, "high": high, **prefixes})
    session.commit()

    return int(result.rowcount)  # type: ignore[attr-defined]


def create_field_search_index(session: Session, column: str) -> None:
    """Build the StreamingDiskANN cosine index of one profile's vectors.

    Commits.

    Args:
        session: Active database session.
        column: One of `PROFILE_VECTORS`.
    """
    session.execute(text(f"SET maintenance_work_mem = '{INDEX_MEMORY}'"))
    session.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS {TABLE}_{column}_idx "
            f"ON {TABLE} USING diskann ({column} vector_cosine_ops)"
        )
    )
    session.execute(text(f"ANALYZE {TABLE}"))
    session.commit()


def index_build_share(session: Session) -> float | None:
    """Read how far a running index build has got, from Postgres's progress view.

    Args:
        session: Active database session (not the one building).

    Returns:
        The share of blocks done, or None when no build reports blocks.
    """
    row = session.execute(
        text(
            "SELECT blocks_done, blocks_total FROM pg_stat_progress_create_index "
            "WHERE blocks_total > 0 LIMIT 1"
        )
    ).first()
    session.commit()  # end the read, so the next one sees fresh numbers

    return row[0] / row[1] if row is not None else None


def search_field_documents(
    session: Session,
    model: str,
    vector: Sequence[float],
    profile: str,
    areas: Sequence[str],
    sources: Sequence[str],
    channels: Sequence[str],
    limit: int,
) -> list[FieldDocument]:
    """Find the documents whose profile is closest to a query vector.

    One profile is one indexed query. `all` takes each profile's `limit` nearest documents
    and ranks them by the sum of their three cosine scores, computed exactly. The filters sit
    in the same query; StreamingDiskANN keeps reading the index until enough documents pass.

    Args:
        session: Active database session.
        model: Embedding model of the vectors.
        vector: The query's unit vector.
        profile: One of `PROFILE_VECTORS`, or `all`.
        areas: Keep documents with any of these areas; empty keeps all.
        sources: Keep documents of these sources; empty keeps all.
        channels: Keep chat threads of these channels; empty keeps all.
        limit: Documents returned.

    Returns:
        The documents, best first.
    """
    filters = ["model = :model"]
    if areas:
        filters.append("areas && CAST(:areas AS text[])")
    if sources:
        filters.append("source_type = ANY(CAST(:sources AS text[]))")
    if channels:
        filters.append("channel = ANY(CAST(:channels AS text[]))")
    columns = PROFILE_VECTORS if profile == "all" else (profile,)
    nearest = " UNION ".join(
        f"(SELECT document_fields_id FROM {TABLE} "
        f"WHERE {column} IS NOT NULL AND {' AND '.join(filters)} "
        f"ORDER BY {column} <=> CAST(:vector AS vector) LIMIT :limit)"
        for column in columns
    )
    score = " + ".join(
        f"coalesce(1 - ({column} <=> CAST(:vector AS vector)), 0)" for column in columns
    )
    statement = text(f"""
        SELECT doc_id, document_id, source_type, channel
        FROM {TABLE} JOIN ({nearest}) AS nearest USING (document_fields_id)
        WHERE model = :model
        ORDER BY {score} DESC
        LIMIT :limit
    """)
    session.execute(text(f"SET LOCAL diskann.query_rescore = {max(RESCORE, limit)}"))
    session.execute(text(f"SET LOCAL diskann.query_search_list_size = {max(SEARCH_LIST, limit)}"))
    rows = session.execute(
        statement,
        {
            "model": model,
            "vector": "[" + ",".join(f"{value:.8f}" for value in vector) + "]",
            "areas": list(areas),
            "sources": list(sources),
            "channels": list(channels),
            "limit": limit,
        },
    )

    return [
        FieldDocument(doc_id=doc_id, document_id=document_id, source_type=source, channel=channel)
        for doc_id, document_id, source, channel in rows
    ]


def field_search_card(
    session: Session, model: str, doc_id: str
) -> tuple[FieldDocument, MinedFields] | None:
    """Read one indexed document with its mined fields.

    Args:
        session: Active database session.
        model: Embedding model of the vectors.
        doc_id: The document's dataset id.

    Returns:
        The document and its mined fields, or None when it isn't indexed.
    """
    row = session.execute(
        text(f"""
            SELECT s.doc_id, s.document_id, s.source_type, s.channel,
                   f.fields, f.thread_kind, f.areas
            FROM {TABLE} s JOIN document_fields f ON f.id = s.document_fields_id
            WHERE s.model = :model AND s.doc_id = :doc_id
            LIMIT 1
        """),
        {"model": model, "doc_id": doc_id},
    ).first()
    if row is None:
        return None
    doc_id, document_id, source, channel, fields, thread_kind, areas = row
    document = FieldDocument(
        doc_id=doc_id, document_id=document_id, source_type=source, channel=channel
    )

    return document, MinedFields(fields=fields, thread_kind=thread_kind, areas=areas)


def field_search_counts(
    session: Session, model: str
) -> tuple[int, Counter[str], Counter[str], Counter[str]]:
    """Count the indexed documents, in total and per area, source and Slack channel.

    Args:
        session: Active database session.
        model: Embedding model of the vectors.

    Returns:
        (documents, per area, per source, per channel).
    """

    def count(query: str) -> Counter[str]:
        """Run a counting query: one key and its count per row.

        Args:
            query: The query, with `:model`.

        Returns:
            The counts by key.
        """
        rows = session.execute(text(query), {"model": model})

        return Counter({key: number for key, number in rows})

    sources = count(f"SELECT source_type, count(*) FROM {TABLE} WHERE model = :model GROUP BY 1")
    channels = count(
        f"SELECT channel, count(*) FROM {TABLE} WHERE model = :model AND channel <> '' GROUP BY 1"
    )
    areas = count(
        f"SELECT area, count(*) FROM {TABLE}, unnest(areas) AS area WHERE model = :model GROUP BY 1"
    )

    return sum(sources.values()), areas, sources, channels
