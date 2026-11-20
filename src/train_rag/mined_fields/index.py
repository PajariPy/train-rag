"""Field search: compare a query with the mined-field profiles of every mined document.

The profiles' unit vectors are held in memory (documents x 3 x 768, float32: about 240 MB
for the ~26k documents mined so far). A query is embedded with the same fine-tuned gte and
scored by cosine similarity with one profile, or the sum of all three; filters on areas,
sources and Slack channels drop documents before ranking. Only mined documents can be
found: the rest of the corpus is reached through the general search.

When a document has rows from several extractors, the one earliest in
`EXTRACTOR_PREFERENCE` is used, so importing the SLM's output (extractor `slm-<run>`)
switches search to it without code changes.
"""

import threading
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sqlalchemy import Engine
from sqlmodel import Session

from train_rag.embedder import Embedder
from train_rag.mined_fields.profiles import PROFILES
from train_rag.models.mined_fields import MinedFields
from train_rag.repository.document_fields import fields_with_vectors
from train_rag.tables import DocumentFieldsRecord, FieldVectorRecord

# Extractor name prefixes, preferred first; within one prefix the newest row wins.
EXTRACTOR_PREFERENCE = ("slm-", "deepseek-")
# Documents a field search returns, best first, for the agent's candidate list.
KEPT = 300
# Profile -> columns of the profile matrix summed for it.
PROFILE_COLUMNS: dict[str, tuple[int, ...]] = {"subject": (0,), "problem": (1,), "names": (2,)}
PROFILE_COLUMNS["all"] = (0, 1, 2)

FieldRow = tuple[DocumentFieldsRecord, FieldVectorRecord]


def preference(extractor: str) -> int:
    """Rank an extractor by `EXTRACTOR_PREFERENCE`; unknown extractors come last.

    Args:
        extractor: The extractor's name.

    Returns:
        0 for the most preferred prefix.
    """
    return next(
        (rank for rank, prefix in enumerate(EXTRACTOR_PREFERENCE) if extractor.startswith(prefix)),
        len(EXTRACTOR_PREFERENCE),
    )


def preferred_rows(rows: Sequence[FieldRow]) -> list[FieldRow]:
    """Keep one row per document: the preferred extractor's, the newest among equals.

    Args:
        rows: Rows of any extractors.

    Returns:
        One row per document, in the order of first appearance.
    """
    best: dict[int, FieldRow] = {}
    for row in rows:
        fields = row[0]
        kept = best.get(fields.document_id)
        if kept is None or (preference(fields.extractor), -fields.created_at.timestamp()) < (
            preference(kept[0].extractor),
            -kept[0].created_at.timestamp(),
        ):
            best[fields.document_id] = row

    return list(best.values())


def unit_or_zero(vector: Sequence[float] | None) -> np.ndarray:
    """A profile vector as float32, zeros when the profile was empty.

    Args:
        vector: The stored vector, or None.

    Returns:
        The vector.
    """
    if vector is None:
        return np.zeros(768, dtype=np.float32)

    return np.asarray(vector, dtype=np.float32)


class FieldIndex:
    """The mined documents' profiles, searchable by a query text."""

    def __init__(self, rows: Sequence[FieldRow], embedder: Embedder) -> None:
        """Hold the preferred row of every mined document.

        Args:
            rows: Mined fields with their profile vectors, of any extractors.
            embedder: The model that embedded the profiles; it embeds the queries.
        """
        kept = preferred_rows(rows)
        self.doc_ids = [fields.doc_id for fields, _ in kept]
        self.document_ids = [fields.document_id for fields, _ in kept]
        self.sources = [fields.source_type for fields, _ in kept]
        self.channels = [fields.channel or "" for fields, _ in kept]
        self.areas = [set(fields.areas) for fields, _ in kept]
        self.mined = [
            MinedFields(fields=fields.fields, thread_kind=fields.thread_kind, areas=fields.areas)
            for fields, _ in kept
        ]
        self.matrix = (
            np.stack(
                [
                    np.stack([unit_or_zero(getattr(vectors, profile)) for profile in PROFILES])
                    for _, vectors in kept
                ]
            )
            if kept
            else np.zeros((0, len(PROFILES), 768), dtype=np.float32)
        )
        self.position = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
        self.embedder = embedder
        self.lock = threading.Lock()  # one model shared by the evaluation's worker threads

    @classmethod
    def load(cls, engine: Engine, embedder: Embedder) -> "FieldIndex":
        """Load every mined document with vectors from the embedder.

        Args:
            engine: Database engine.
            embedder: The model the profile vectors come from.

        Returns:
            The index.
        """
        with Session(engine) as session:
            rows = fields_with_vectors(session, None, embedder.name)

        return cls(rows, embedder)

    def __len__(self) -> int:
        """Count the mined documents.

        Returns:
            Their number.
        """
        return len(self.doc_ids)

    def area_counts(self) -> Counter[str]:
        """Count the documents per area.

        Returns:
            Documents per area.
        """
        return Counter(area for found in self.areas for area in found)

    def source_counts(self) -> Counter[str]:
        """Count the documents per source.

        Returns:
            Documents per source.
        """
        return Counter(self.sources)

    def channel_counts(self) -> Counter[str]:
        """Count the chat threads per Slack channel.

        Returns:
            Threads per channel.
        """
        return Counter(channel for channel in self.channels if channel)

    def search(
        self,
        query: str,
        profile: str,
        areas: Sequence[str] = (),
        sources: Sequence[str] = (),
        channels: Sequence[str] = (),
        limit: int = KEPT,
    ) -> list[int]:
        """Rank the mined documents by how close a profile is to the query.

        Args:
            query: The search text.
            profile: `subject`, `problem`, `names` or `all` (the three summed).
            areas: Keep documents with any of these areas; empty keeps all.
            sources: Keep documents of these sources; empty keeps all.
            channels: Keep chat threads of these channels; empty keeps all.
            limit: Documents returned.

        Returns:
            Positions in the index, best first.
        """
        with self.lock:
            vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        vector /= np.linalg.norm(vector) + 1e-9
        columns = PROFILE_COLUMNS.get(profile, PROFILE_COLUMNS["all"])
        scores = sum(self.matrix[:, column] @ vector for column in columns)
        scores = np.asarray(scores, dtype=np.float32)
        if areas:
            allowed = set(areas)
            scores[[not found & allowed for found in self.areas]] = -np.inf
        if sources:
            scores[[source not in sources for source in self.sources]] = -np.inf
        if channels:
            scores[[channel not in channels for channel in self.channels]] = -np.inf

        return [int(index) for index in np.argsort(-scores)[:limit] if np.isfinite(scores[index])]
