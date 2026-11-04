"""Import the downloaded dataset into Postgres."""

from collections.abc import Sequence
from itertools import batched
from pathlib import Path

from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import create_tables, get_engine
from train_rag.models import Artifact, Question
from train_rag.repository.document import count_documents_by_source, insert_documents
from train_rag.repository.question import count_questions, insert_questions
from train_rag.tables import DocumentRecord, QuestionRecord

BATCH_SIZE = 2000
NUL = "\x00"
NUL_ESCAPE = "\\x00"


def escape_nul(text: str) -> str:
    r"""Replace NUL characters, which Postgres text cannot store, with a visible escape.

    A few documents mention a NUL delimiter literally, so `\x00` keeps their meaning.

    Args:
        text: Text to store.

    Returns:
        The text with every NUL written as `\x00`.
    """
    return text.replace(NUL, NUL_ESCAPE)


def to_document_record(dataset: str, artifact: Artifact) -> DocumentRecord:
    """Convert an artifact into a documents row.

    Args:
        dataset: Name of the dataset the artifact belongs to.
        artifact: Loaded artifact with provenance.

    Returns:
        The row to store.

    Raises:
        ValueError: If the artifact has no provenance.
    """
    if artifact.source_file is None or artifact.row_index is None:
        raise ValueError(f"Artifact {artifact.id} has no source file or row index")

    return DocumentRecord(
        doc_id=artifact.id,
        dataset=dataset,
        source_type=str(artifact.source_type),
        title=escape_nul(artifact.title),
        content=escape_nul(artifact.content),
        char_count=len(artifact.content),
        byte_size=len(artifact.content.encode()),
        source_file=artifact.source_file,
        row_index=artifact.row_index,
    )


def to_question_record(dataset: str, question: Question) -> QuestionRecord:
    """Convert a benchmark question into a questions row.

    Args:
        dataset: Name of the dataset the question belongs to.
        question: Loaded question.

    Returns:
        The row to store.
    """
    return QuestionRecord(
        id=question.id,
        dataset=dataset,
        question_type=str(question.question_type),
        text=question.text,
        gold_answer=question.gold_answer,
        answer_facts=question.answer_facts,
        source_types=[str(source) for source in question.source_types],
        expected_doc_ids=question.expected_doc_ids,
    )


def import_dataset(session: Session, dataset: str, artifacts: Sequence[Artifact]) -> None:
    """Store artifacts in batches, committing after each batch.

    Args:
        session: Active database session.
        dataset: Name of the dataset.
        artifacts: Artifacts to store.
    """
    for batch in batched(artifacts, BATCH_SIZE, strict=False):
        insert_documents(session, [to_document_record(dataset, artifact) for artifact in batch])
        session.commit()


def import_corpus(data_dir: Path) -> None:
    """Load the dataset from disk and import it into Postgres.

    Rows that are already stored are skipped, so the import can be rerun.

    Args:
        data_dir: Directory the dataset was downloaded to.
    """
    loader = EnterpriseRagBenchLoader(data_dir)
    corpus = loader.load()
    engine = get_engine()
    create_tables(engine)

    with Session(engine) as session:
        import_dataset(session, loader.name, corpus.artifacts)
        insert_questions(
            session, [to_question_record(loader.name, question) for question in corpus.questions]
        )
        session.commit()

        print(f"Documents per source: {count_documents_by_source(session, loader.name)}")
        print(f"Questions: {count_questions(session, loader.name)}")
