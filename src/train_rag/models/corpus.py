"""Everything a data loader loads from a data source."""

from pydantic import BaseModel, Field

from train_rag.models.artifact import Artifact
from train_rag.models.question import Question


class Corpus(BaseModel):
    """Artifacts and evaluation questions from one source."""

    source: str
    artifacts: list[Artifact] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
