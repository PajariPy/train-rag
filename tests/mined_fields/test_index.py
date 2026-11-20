"""Tests for field search: profiles, filters and the extractor preference."""

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np

from train_rag.embedder import Embedder
from train_rag.mined_fields.index import FieldIndex, preferred_rows
from train_rag.tables import DocumentFieldsRecord, FieldVectorRecord

DIMENSIONS = 768


def axis(index: int) -> list[float]:
    """A unit vector along one axis.

    Args:
        index: The axis.

    Returns:
        The vector.
    """
    vector = np.zeros(DIMENSIONS)
    vector[index] = 1.0

    return vector.tolist()


class AxisEmbedder(Embedder):
    """Embeds a query named "axis N" as the unit vector along axis N."""

    name = "axis"
    dimensions = DIMENSIONS
    max_tokens = 10

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed each text.

        Args:
            texts: Queries like "axis 3".

        Returns:
            One unit vector per text.
        """
        return [axis(int(text.split()[1])) for text in texts]


def row(
    document_id: int,
    vectors: tuple[int | None, int | None, int | None],
    areas: list[str],
    source: str = "slack",
    channel: str | None = None,
    extractor: str = "deepseek-x",
    day: int = 1,
) -> tuple[DocumentFieldsRecord, FieldVectorRecord]:
    """Build one mined document with its profile vectors.

    Args:
        document_id: Row id; the dataset id is "d<row id>".
        vectors: The axis of the subject, problem and names vectors (None: empty).
        areas: Its areas.
        source: Its source.
        channel: Its Slack channel.
        extractor: Who mined it.
        day: Day of the month it was stored.

    Returns:
        The fields and vectors rows.
    """
    fields = DocumentFieldsRecord(
        id=document_id,
        document_id=document_id,
        doc_id=f"d{document_id}",
        source_type=source,
        extractor=extractor,
        raw={},
        fields={"topic": [f"topic {document_id}"]},
        areas=areas,
        channel=channel,
        created_at=datetime(2026, 11, day, tzinfo=UTC),
    )
    subject, problem, names = (None if value is None else axis(value) for value in vectors)
    profile = FieldVectorRecord(
        document_fields_id=document_id, model="axis", subject=subject, problem=problem, names=names
    )

    return fields, profile


def index() -> FieldIndex:
    """Build a small index of four documents.

    Returns:
        The index.
    """
    rows = [
        row(1, (0, 1, None), ["billing and pricing"], channel="billing"),
        row(2, (0, 2, 3), ["customer support"], source="gmail"),
        row(3, (5, 1, 3), ["billing and pricing", "legal"], channel="legal"),
        row(4, (None, None, None), ["other"]),
    ]

    return FieldIndex(rows, AxisEmbedder())


def doc_ids(found: list[int], field_index: FieldIndex) -> list[str]:
    """Turn index positions into dataset ids.

    Args:
        found: Positions, best first.
        field_index: The index.

    Returns:
        The dataset ids.
    """
    return [field_index.doc_ids[position] for position in found]


def test_profile_picks_the_compared_vector() -> None:
    """A subject search matches subjects only; an empty profile scores zero."""
    field_index = index()
    found = field_index.search("axis 0", "subject", limit=2)
    assert doc_ids(found, field_index) == ["d1", "d2"]
    assert doc_ids(field_index.search("axis 3", "names", limit=2), field_index) == ["d2", "d3"]


def test_all_sums_the_three_profiles() -> None:
    """With "all", a document matching on two profiles beats one matching on one."""
    rows = [row(1, (7, None, None), []), row(2, (7, 7, None), []), row(3, (None, None, 7), [])]
    field_index = FieldIndex(rows, AxisEmbedder())
    assert doc_ids(field_index.search("axis 7", "all"), field_index)[0] == "d2"
    assert doc_ids(field_index.search("axis 7", "names", limit=1), field_index) == ["d3"]


def test_filters_drop_documents() -> None:
    """Areas match any of the document's areas; sources and channels must match exactly."""
    field_index = index()
    by_area = field_index.search("axis 0", "subject", areas=["legal"])
    assert doc_ids(by_area, field_index) == ["d3"]
    by_source = field_index.search("axis 0", "subject", sources=["gmail"])
    assert doc_ids(by_source, field_index) == ["d2"]
    by_channel = field_index.search("axis 0", "subject", channels=["billing", "legal"])
    assert set(doc_ids(by_channel, field_index)) == {"d1", "d3"}


def test_counts_for_the_prompt() -> None:
    """Areas, sources and channels are counted per document."""
    field_index = index()
    assert field_index.area_counts()["billing and pricing"] == 2
    assert field_index.source_counts() == {"slack": 3, "gmail": 1}
    assert field_index.channel_counts() == {"billing": 1, "legal": 1}


def test_slm_rows_are_preferred_then_the_newest() -> None:
    """An slm- row beats a deepseek- row of the same document; among equals the newest wins."""
    rows = [
        row(1, (0, 0, 0), [], extractor="deepseek-a", day=9),
        row(1, (1, 1, 1), [], extractor="slm-run1", day=1),
        row(1, (2, 2, 2), [], extractor="slm-run2", day=5),
        row(2, (0, 0, 0), [], extractor="deepseek-a"),
    ]
    kept = preferred_rows(rows)
    assert [(fields.document_id, fields.extractor) for fields, _ in kept] == [
        (1, "slm-run2"),
        (2, "deepseek-a"),
    ]
