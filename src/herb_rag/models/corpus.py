"""Everything a connector loads from a data source."""

from pydantic import BaseModel, Field

from herb_rag.models.artifact import Artifact
from herb_rag.models.people import Customer, Employee
from herb_rag.models.question import Question


class Corpus(BaseModel):
    """Artifacts, evaluation questions and directory data from one source."""

    source: str
    artifacts: list[Artifact] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    employees: list[Employee] = Field(default_factory=list)
    customers: list[Customer] = Field(default_factory=list)
