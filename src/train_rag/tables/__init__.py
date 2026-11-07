"""Database table models."""

from train_rag.tables.document import DocumentRecord
from train_rag.tables.embedding import EMBEDDING_TABLES, EmbeddingRecord
from train_rag.tables.evaluation import EvalResultRecord, EvalRunRecord
from train_rag.tables.inferred_attendees import InferredAttendeesRecord
from train_rag.tables.inferred_date import InferredDateRecord
from train_rag.tables.inferred_messages import InferredMessagesRecord
from train_rag.tables.parsed_document import ParsedDocumentRecord
from train_rag.tables.pipeline_run import PipelineRunRecord
from train_rag.tables.question import QuestionRecord

__all__ = [
    "EMBEDDING_TABLES",
    "DocumentRecord",
    "EmbeddingRecord",
    "EvalResultRecord",
    "EvalRunRecord",
    "InferredAttendeesRecord",
    "InferredDateRecord",
    "InferredMessagesRecord",
    "ParsedDocumentRecord",
    "PipelineRunRecord",
    "QuestionRecord",
]
