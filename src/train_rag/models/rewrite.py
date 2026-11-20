"""Gemini's search queries for one question, appended to it for keyword search."""

from pydantic import BaseModel


class Rewrites(BaseModel):
    """The rewritten search queries of one question, keyword query first."""

    queries: list[str]
