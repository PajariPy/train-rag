"""Embed the profiles of mined fields: the vectors field search compares queries with."""

import numpy as np
from sqlalchemy import Engine
from sqlmodel import Session

from train_rag.embedder import Embedder
from train_rag.mined_fields.profiles import PROFILES, profiles
from train_rag.models.mined_fields import MinedFields
from train_rag.progress.recorder import RunProgress
from train_rag.repository.document_fields import (
    count_fields_without_vectors,
    fields_without_vectors,
    upsert_field_vectors,
)
from train_rag.tables import DocumentFieldsRecord, FieldVectorRecord

BATCH_SIZE = 512


def profile_vectors(
    records: list[DocumentFieldsRecord], embedder: Embedder
) -> list[FieldVectorRecord]:
    """Embed the three profiles of each record; an empty profile gets no vector.

    Args:
        records: Stored mined fields.
        embedder: The model that embeds the profiles and, later, the queries.

    Returns:
        One vector row per record.
    """
    texts = [
        getattr(profiles(MinedFields(fields=record.fields)), profile)
        for record in records
        for profile in PROFILES
    ]
    filled = [index for index, text in enumerate(texts) if text]
    embedded = embedder.embed([texts[index] for index in filled]) if filled else []
    vectors: list[list[float] | None] = [None] * len(texts)
    for index, vector in zip(filled, embedded, strict=True):
        vectors[index] = (np.asarray(vector) / (np.linalg.norm(vector) + 1e-9)).tolist()

    return [
        FieldVectorRecord(
            document_fields_id=record.id or 0,
            model=embedder.name,
            subject=vectors[3 * row],
            problem=vectors[3 * row + 1],
            names=vectors[3 * row + 2],
        )
        for row, record in enumerate(records)
    ]


def embed_field_profiles(engine: Engine, embedder: Embedder) -> int:
    """Embed every stored mined fields row that has no vectors from this model yet.

    Args:
        engine: Database engine.
        embedder: The embedding model.

    Returns:
        The number of rows embedded.
    """
    with Session(engine) as session:
        total = count_fields_without_vectors(session, embedder.name)
    done = 0
    title = f"embed mined-field profiles with {embedder.name}"
    with RunProgress(engine, title, total) as progress, Session(engine) as session:
        while records := fields_without_vectors(session, embedder.name, BATCH_SIZE):
            upsert_field_vectors(session, profile_vectors(records, embedder))
            session.commit()
            done += len(records)
            progress.advance("embedding", len(records))

    return done
