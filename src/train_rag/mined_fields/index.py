"""Field search: compare a query with the mined-field profiles of every mined document.

A query is embedded with the same fine-tuned gte and scored by cosine similarity with one
profile (subject, problem, names), or the sum of all three; filters on areas, sources and
Slack channels drop documents before ranking. Only mined documents can be found: the rest
of the corpus is reached through the general search.

`FieldSearch` is what the agent uses. Two implementations:
- `StoredFieldIndex` (`mined_fields.stored`): the whole corpus, searched in Postgres with
  StreamingDiskANN indexes; nothing is loaded into memory;
- `FieldIndex` (here): a small set held in numpy, exact; for tests and experiments.

When a document has rows from several extractors, the one earliest in
`EXTRACTOR_PREFERENCE` is used, so importing the SLM's output (extractor `slm-<run>`)
switches search to it without code changes.
"""

import threading
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

from train_rag.embedder import Embedder
from train_rag.mined_fields.profiles import PROFILES
from train_rag.models.mined_fields import FieldDocument, MinedFields
from train_rag.tables import DocumentFieldsRecord
from train_rag.tables.field_vector import PROFILE_DIMENSIONS

# Extractor name prefixes, preferred first; within one prefix the newest row wins.
EXTRACTOR_PREFERENCE = ("slm-", "deepseek-")
# Documents a field search returns, best first, for the agent's candidate list.
KEPT = 300
# Profile -> columns of the profile matrix summed for it.
PROFILE_COLUMNS: dict[str, tuple[int, ...]] = {"subject": (0,), "problem": (1,), "names": (2,)}
PROFILE_COLUMNS["all"] = (0, 1, 2)

# A mined document's fields and its (3, dimensions) profile matrix.
FieldRow = tuple[DocumentFieldsRecord, npt.NDArray[np.float32]]


class FieldSearch(Protocol):
    """What the agent needs from field search."""

    def __len__(self) -> int:
        """Count the mined documents.

        Returns:
            Their number.
        """
        ...

    def area_counts(self) -> Counter[str]:
        """Count the documents per area.

        Returns:
            Documents per area.
        """
        ...

    def source_counts(self) -> Counter[str]:
        """Count the documents per source.

        Returns:
            Documents per source.
        """
        ...

    def channel_counts(self) -> Counter[str]:
        """Count the chat threads per Slack channel.

        Returns:
            Threads per channel.
        """
        ...

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
        ...

    def card(self, doc_id: str) -> tuple[FieldDocument, MinedFields] | None:
        """Look up a document's mined fields for its result card.

        Args:
            doc_id: The document.

        Returns:
            The document and its mined fields, or None when it wasn't mined.
        """
        ...


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


class FieldIndex:
    """A small set of mined documents held in memory, searched exactly."""

    def __init__(self, rows: Sequence[FieldRow], embedder: Embedder) -> None:
        """Hold the preferred row of every mined document.

        Args:
            rows: Mined fields with their (3, dimensions) profile matrices, any extractors.
            embedder: The model that embedded the profiles; it embeds the queries.
        """
        kept = preferred_rows(rows)
        self.documents = [
            FieldDocument(
                doc_id=fields.doc_id,
                document_id=fields.document_id,
                source_type=fields.source_type,
                channel=fields.channel or "",
            )
            for fields, _ in kept
        ]
        self.areas = [frozenset(fields.areas) for fields, _ in kept]
        self.mined = [
            MinedFields(fields=fields.fields, thread_kind=fields.thread_kind, areas=fields.areas)
            for fields, _ in kept
        ]
        self.matrix = (
            np.stack([matrix for _, matrix in kept]).astype(np.float32)
            if kept
            else np.zeros((0, len(PROFILES), PROFILE_DIMENSIONS), dtype=np.float32)
        )
        self.position = {document.doc_id: index for index, document in enumerate(self.documents)}
        self.embedder = embedder
        self.lock = threading.Lock()  # one model shared by the evaluation's worker threads

    def __len__(self) -> int:
        """Count the mined documents.

        Returns:
            Their number.
        """
        return len(self.documents)

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
        return Counter(document.source_type for document in self.documents)

    def channel_counts(self) -> Counter[str]:
        """Count the chat threads per Slack channel.

        Returns:
            Threads per channel.
        """
        return Counter(document.channel for document in self.documents if document.channel)

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

        columns = PROFILE_COLUMNS.get(profile, PROFILE_COLUMNS["all"])
        scores = sum(self.matrix[:, column] @ vector for column in columns)
        scores = np.asarray(scores, dtype=np.float32)

        if areas:
            allowed = set(areas)
            scores[[not found & allowed for found in self.areas]] = -np.inf
        if sources:
            scores[[document.source_type not in sources for document in self.documents]] = -np.inf
        if channels:
            scores[[document.channel not in channels for document in self.documents]] = -np.inf

        ranked = [int(index) for index in np.argsort(-scores)[:limit] if np.isfinite(scores[index])]

        return [self.documents[index] for index in ranked]

    def card(self, doc_id: str) -> tuple[FieldDocument, MinedFields] | None:
        """Look up a document's mined fields for its result card.

        Args:
            doc_id: The document.

        Returns:
            The document and its mined fields, or None when it wasn't mined.
        """
        position = self.position.get(doc_id)
        if position is None:
            return None

        return self.documents[position], self.mined[position]
