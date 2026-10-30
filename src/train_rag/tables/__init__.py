"""Database table models."""

from train_rag.tables.document import DocumentRecord
from train_rag.tables.inferred_attendees import InferredAttendeesRecord
from train_rag.tables.inferred_date import InferredDateRecord
from train_rag.tables.question import QuestionRecord

__all__ = [
    "DocumentRecord",
    "InferredAttendeesRecord",
    "InferredDateRecord",
    "QuestionRecord",
]
