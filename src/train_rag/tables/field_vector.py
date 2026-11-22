"""Profile vectors of mined fields: what field search compares a query with."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, SQLModel

PROFILE_DIMENSIONS = 768


class FieldVectorRecord(SQLModel, table=True):
    """The three profile vectors of one document's mined fields, from one embedding model.

    subject = topic + concepts, problem = bug + cause + fix + question, names = every name
    field. A profile with nothing mined has no vector (NULL), so it matches nothing.
    """

    __tablename__ = "document_field_vectors"

    document_fields_id: int = Field(foreign_key="document_fields.id", primary_key=True)
    model: str = Field(primary_key=True)
    subject: list[float] | None = Field(
        default=None, sa_column=Column(Vector(PROFILE_DIMENSIONS), nullable=True)
    )
    problem: list[float] | None = Field(
        default=None, sa_column=Column(Vector(PROFILE_DIMENSIONS), nullable=True)
    )
    names: list[float] | None = Field(
        default=None, sa_column=Column(Vector(PROFILE_DIMENSIONS), nullable=True)
    )
