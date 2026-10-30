"""Meeting dates estimated by an LLM judge, stored apart from the raw documents."""

from datetime import date

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


class InferredDateRecord(SQLModel, table=True):
    """The date one meeting was held, as estimated from clues in its text.

    `method` names the judge that produced it, so a rerun with another model or prompt
    can replace its own rows.
    """

    __tablename__ = "inferred_dates"

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", unique=True, index=True)
    held_on: date
    precision: str
    evidence: str = Field(sa_column=Column(Text, nullable=False))
    year_in_evidence: bool
    method: str = Field(index=True)
