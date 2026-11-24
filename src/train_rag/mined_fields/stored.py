"""Field search over the whole corpus in Postgres, and the command that builds its table.

`build_field_search` copies the preferred fields row of every mined document with its
profile vectors into `field_search_vectors` and builds one StreamingDiskANN index per
profile (`train-rag index-fields`, after `embed-fields`). `StoredFieldIndex` then searches
there: nothing is loaded into memory, so it starts in seconds at any corpus size.
"""

import threading
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sqlalchemy import Engine
from sqlmodel import Session

from train_rag.embedder import Embedder
from train_rag.mined_fields.index import EXTRACTOR_PREFERENCE, KEPT
from train_rag.models.mined_fields import FieldDocument, MinedFields
from train_rag.progress.recorder import RunProgress
from train_rag.repository.field_search import (
    PROFILE_VECTORS,
    clear_field_search,
    create_field_search_index,
    document_id_bounds,
    field_search_card,
    field_search_counts,
    fill_field_search,
    index_build_share,
    search_field_documents,
)

# Document row ids copied per statement, for the progress bar.
FILL_RANGE = 25_000
# Seconds between progress updates while an index builds (one statement, ~40 min each).
INDEX_REPORT_SECONDS = 30


def build_index(engine: Engine, profile: str, progress: RunProgress) -> None:
    """Build one profile's index in a worker thread, reporting its progress meanwhile.

    A build is one long statement; without updates the progress page would show the run
    as stopped after 5 minutes.

    Args:
        engine: Database engine.
        profile: A key of `PROFILE_VECTORS`.
        progress: The run's progress record (used from this thread only).
    """

    def create() -> None:
        """Build the index on its own connection."""
        with Session(engine) as session:
            create_field_search_index(session, profile)

    with ThreadPoolExecutor(max_workers=1) as pool, Session(engine) as session:
        build = pool.submit(create)
        while not build.done():
            share = index_build_share(session)
            graph = f"{share:.0%} of the graph" if share is not None else "starting"
            progress.advance(f"index {profile}", 0, note=f"index {profile}: {graph}")
            try:
                build.result(timeout=INDEX_REPORT_SECONDS)
            except TimeoutError:
                continue
        build.result()  # raise the build's error, if any


def build_field_search(engine: Engine, model: str, workflow: str | None = None) -> int:
    """Rebuild field search's table and indexes for one embedding model.

    Args:
        engine: Database engine.
        model: Embedding model of the profile vectors.
        workflow: Key of a workflow whose "Index the fields" step this is, or None.

    Returns:
        Documents indexed.
    """
    with Session(engine) as session:
        low, high = document_id_bounds(session)
        clear_field_search(session, model)
    starts = list(range(low, high + 1, FILL_RANGE))
    total = len(starts) + len(PROFILE_VECTORS)
    step = "Index the fields" if workflow else None
    written = 0

    title = f"index mined fields for search ({model})"
    with (
        RunProgress(engine, title, total, unit="steps", workflow=workflow, step=step) as progress,
        Session(engine) as session,
    ):
        for start in starts:
            written += fill_field_search(
                session, model, EXTRACTOR_PREFERENCE, start, start + FILL_RANGE
            )
            progress.advance("copying", 1, note=f"{written:,} documents")
        for profile in PROFILE_VECTORS:
            build_index(engine, profile, progress)
            progress.advance(f"index {profile}", 1, note=f"index {profile}: done")

    return written


class StoredFieldIndex:
    """Field search in Postgres: the whole corpus, StreamingDiskANN, nothing in memory."""

    def __init__(self, engine: Engine, embedder: Embedder) -> None:
        """Read the counts the agent's instructions show.

        Args:
            engine: Database engine.
            embedder: The model that embedded the profiles; it embeds the queries.
        """
        self.engine = engine
        self.embedder = embedder
        self.lock = threading.Lock()  # one model shared by the evaluation's worker threads
        with Session(engine) as session:
            self.size, self.areas, self.sources, self.channels = field_search_counts(
                session, embedder.name
            )
        if not self.size:
            raise LookupError(f"no field search rows for {embedder.name}: run index-fields")

    def __len__(self) -> int:
        """Count the mined documents.

        Returns:
            Their number.
        """
        return self.size

    def area_counts(self) -> Counter[str]:
        """Count the documents per area.

        Returns:
            Documents per area.
        """
        return self.areas

    def source_counts(self) -> Counter[str]:
        """Count the documents per source.

        Returns:
            Documents per source.
        """
        return self.sources

    def channel_counts(self) -> Counter[str]:
        """Count the chat threads per Slack channel.

        Returns:
            Threads per channel.
        """
        return self.channels

    def search(
        self,
        query: str,
        profile: str,
        areas: Sequence[str] = (),
        sources: Sequence[str] = (),
        channels: Sequence[str] = (),
        limit: int = KEPT,
    ) -> list[FieldDocument]:
        """Rank the mined documents by how close a profile is to the query.

        Args:
            query: The search text.
            profile: `subject`, `problem`, `names` or `all` (the three summed).
            areas: Keep documents with any of these areas; empty keeps all.
            sources: Keep documents of these sources; empty keeps all.
            channels: Keep chat threads of these channels; empty keeps all.
            limit: Documents returned.

        Returns:
            The documents, best first.
        """
        with self.lock:
            vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        vector /= np.linalg.norm(vector) + 1e-9
        profile = profile if profile in PROFILE_VECTORS else "all"

        with Session(self.engine) as session:
            return search_field_documents(
                session,
                self.embedder.name,
                vector.tolist(),
                profile,
                areas,
                sources,
                channels,
                limit,
            )

    def card(self, doc_id: str) -> tuple[FieldDocument, MinedFields] | None:
        """Look up a document's mined fields for its result card.

        Args:
            doc_id: The document.

        Returns:
            The document and its mined fields, or None when it wasn't mined.
        """
        with Session(self.engine) as session:
            return field_search_card(session, self.embedder.name, doc_id)
