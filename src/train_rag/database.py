"""Postgres connection settings and schema creation."""

import os

from sqlalchemy import Engine
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


def create_tables(engine: Engine) -> None:
    """Create every table that does not exist yet.

    Args:
        engine: Engine of the target database.
    """
    SQLModel.metadata.create_all(engine)
