"""Postgres connection settings and schema creation."""

import os

from sqlalchemy import Engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel, create_engine

from train_rag import tables  # noqa: F401  (registers table models on SQLModel.metadata)

DEFAULT_DATABASE_URL = "postgresql+psycopg://train_rag:train_rag@localhost:5434/train_rag"


def database_url() -> str:
    """Resolve the database URL.

    Returns:
        The `DATABASE_URL` environment variable, or the local docker-compose database.
    """
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine() -> Engine:
    """Create an engine for the configured database.

    Returns:
        A SQLAlchemy engine.
    """
    return create_engine(database_url())


def get_async_engine() -> AsyncEngine:
    """Create an async engine for the configured database, for the retrieval pipeline.

    Returns:
        A SQLAlchemy async engine on psycopg's async driver.
    """
    return create_async_engine(database_url())


def create_tables(engine: Engine) -> None:
    """Create the vector extensions and every table that does not exist yet.

    `vectorscale` brings pgvector (the `vector` type) and the StreamingDiskANN index used
    by the embedding tables.

    Args:
        engine: Engine of the target database.
    """
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE"))
    SQLModel.metadata.create_all(engine)
    with engine.begin() as connection:
        # Added after eval_results was first created; create_all doesn't add columns.
        connection.execute(
            text(
                "ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS cited_doc_ids JSONB "
                "NOT NULL DEFAULT '[]'"
            )
        )
