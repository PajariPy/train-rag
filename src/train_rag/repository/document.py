"""Queries for the documents table."""

from collections.abc import Iterator, Sequence

from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert
from sqlmodel import Session, col, select

from train_rag.models import Artifact, SourceType
from train_rag.models.document_size import DocumentSize
from train_rag.tables import DocumentRecord


def insert_documents(session: Session, documents: Sequence[DocumentRecord]) -> None:
    """Insert documents, skipping rows already imported from the same file and row.

    Args:
        session: Active database session.
        documents: Documents to insert.
    """
    if not documents:
        return

    rows = [document.model_dump(exclude={"id"}) for document in documents]
    statement = insert(DocumentRecord).on_conflict_do_nothing(
        index_elements=["dataset", "source_file", "row_index"]
    )
    session.execute(statement, rows)


def count_documents_by_source(session: Session, dataset: str) -> dict[str, int]:
    """Count stored documents per source type.

    Args:
        session: Active database session.
        dataset: Dataset to count documents of.

    Returns:
        Document counts keyed by source type.
    """
    statement = (
        select(DocumentRecord.source_type, func.count())
        .where(DocumentRecord.dataset == dataset)
        .group_by(DocumentRecord.source_type)
    )

    return {source_type: count for source_type, count in session.exec(statement)}


def document_sizes(session: Session, dataset: str, chars_per_line: int) -> list[DocumentSize]:
    """Measure every document in characters and in lines of a printed page.

    Each line of the text (escaped line breaks, backslash-n, count as line breaks too)
    takes at least one printed line and wraps every `chars_per_line` characters. Postgres
    does the counting, so the text never leaves the database.

    Args:
        session: Active database session.
        dataset: Dataset to measure.
        chars_per_line: Characters that fit on one printed line.

    Returns:
        One size per document, in row order.
    """
    statement = text(
        r"""
        SELECT id, source_type, char_count,
            (SELECT sum(greatest(1, ceil(length(line)::numeric / :chars_per_line)))
             FROM string_to_table(replace(content, '\n', E'\n'), E'\n') AS line) AS lines
        FROM documents
        WHERE dataset = :dataset
        ORDER BY id
        """
    )
    rows = session.connection().execute(
        statement, {"dataset": dataset, "chars_per_line": chars_per_line}
    )

    return [
        DocumentSize(
            document_id=document_id, source_type=source_type, chars=chars, lines=int(lines or 0)
        )
        for document_id, source_type, chars, lines in rows
    ]


def get_document(session: Session, document_id: int) -> DocumentRecord | None:
    """Get the raw document a parsed document links to.

    Args:
        session: Active database session.
        document_id: Row id of the document.

    Returns:
        The document, or None when there is no such row.
    """
    return session.get(DocumentRecord, document_id)


def load_artifacts(
    session: Session, dataset: str, source_type: SourceType, limit: int | None = None
) -> list[Artifact]:
    """Load stored documents of one source as artifacts linked to their rows.

    Args:
        session: Active database session.
        dataset: Dataset to load documents of.
        source_type: Source to load.
        limit: Maximum number of documents, or None for all.

    Returns:
        Artifacts in row order, with `document_id` set.
    """
    statement = (
        select(DocumentRecord)
        .where(DocumentRecord.dataset == dataset, DocumentRecord.source_type == source_type)
        .order_by(col(DocumentRecord.id))
        .limit(limit)
    )

    return [to_artifact(document) for document in session.exec(statement)]


def document_ids(session: Session, dataset: str, source_type: SourceType) -> list[int]:
    """List the row ids of one source's documents.

    Args:
        session: Active database session.
        dataset: Dataset to list documents of.
        source_type: Source to list.

    Returns:
        The row ids, lowest first.
    """
    statement = (
        select(DocumentRecord.id)
        .where(DocumentRecord.dataset == dataset, DocumentRecord.source_type == source_type)
        .order_by(col(DocumentRecord.id))
    )

    return [document_id for document_id in session.exec(statement) if document_id is not None]


def document_lengths(
    session: Session, dataset: str, source_type: SourceType
) -> list[tuple[int, int]]:
    """List one source's documents with their length, to plan and time a run.

    Args:
        session: Active database session.
        dataset: Dataset to list documents of.
        source_type: Source to list.

    Returns:
        (row id, characters) per document, lowest row id first.
    """
    statement = (
        select(DocumentRecord.id, DocumentRecord.char_count)
        .where(DocumentRecord.dataset == dataset, DocumentRecord.source_type == source_type)
        .order_by(col(DocumentRecord.id))
    )

    return [(document_id, chars) for document_id, chars in session.exec(statement) if document_id]


def bracketed_documents(
    session: Session, dataset: str, batch: int = 2_000
) -> Iterator[tuple[int, SourceType, str]]:
    """Stream the documents whose text starts with `[` and ends with `]`.

    These may be stored as a Python list literal (see `decode_string_list`); the caller
    decides which really are.

    Args:
        session: Active database session.
        dataset: Dataset to search.
        batch: Rows fetched at a time, so the texts never all sit in memory.

    Yields:
        (row id, source, text) of each such document.
    """
    trimmed = func.btrim(DocumentRecord.content)
    statement = (
        select(DocumentRecord.id, DocumentRecord.source_type, DocumentRecord.content)
        .where(
            DocumentRecord.dataset == dataset,
            func.left(trimmed, 1) == "[",
            func.right(trimmed, 1) == "]",
        )
        .order_by(col(DocumentRecord.id))
        .execution_options(yield_per=batch)
    )
    for document_id, source_type, content in session.exec(statement):
        if document_id is not None:
            yield document_id, SourceType(source_type), content


def sample_document_ids(session: Session, dataset: str, size: int, seed: str) -> set[int]:
    """Pick a random but repeatable sample of documents from all sources.

    Documents are ordered by a hash of their dataset id and the seed, so the same seed
    always picks the same documents.

    Args:
        session: Active database session.
        dataset: Dataset to sample from.
        size: Number of documents.
        seed: Text mixed into the hash.

    Returns:
        The row ids of the sample.
    """
    statement = (
        select(DocumentRecord.id)
        .where(DocumentRecord.dataset == dataset)
        .order_by(func.md5(func.concat(DocumentRecord.doc_id, seed)))
        .limit(size)
    )

    return {document_id for document_id in session.exec(statement) if document_id is not None}


def ids_of_doc_ids(session: Session, dataset: str, doc_ids: Sequence[str]) -> set[int]:
    """Find the row ids of documents by their dataset ids.

    Args:
        session: Active database session.
        dataset: Dataset of the documents.
        doc_ids: Dataset ids, e.g. the ones benchmark questions cite.

    Returns:
        The row ids of the documents found.
    """
    statement = select(DocumentRecord.id).where(
        DocumentRecord.dataset == dataset, col(DocumentRecord.doc_id).in_(doc_ids)
    )

    return {document_id for document_id in session.exec(statement) if document_id is not None}


def load_documents(session: Session, ids: Sequence[int]) -> list[DocumentRecord]:
    """Load documents by row id.

    Args:
        session: Active database session.
        ids: Row ids.

    Returns:
        The documents, lowest row id first.
    """
    statement = (
        select(DocumentRecord)
        .where(col(DocumentRecord.id).in_(ids))
        .order_by(col(DocumentRecord.id))
    )

    return list(session.exec(statement))


def sample_artifacts(
    session: Session, dataset: str, source_type: SourceType, size: int
) -> list[Artifact]:
    """Load a random sample of stored documents of one source.

    Args:
        session: Active database session.
        dataset: Dataset to sample from.
        source_type: Source to sample.
        size: Number of documents to sample.

    Returns:
        Up to `size` artifacts, with `document_id` set.
    """
    statement = (
        select(DocumentRecord)
        .where(DocumentRecord.dataset == dataset, DocumentRecord.source_type == source_type)
        .order_by(func.random())
        .limit(size)
    )

    return [to_artifact(document) for document in session.exec(statement)]


def to_artifact(document: DocumentRecord) -> Artifact:
    """Convert a documents row into an artifact that keeps the row id.

    Args:
        document: Stored document.

    Returns:
        The artifact, ready to be parsed.
    """
    return Artifact(
        id=document.doc_id,
        source_type=SourceType(document.source_type),
        title=document.title,
        content=document.content,
        source_file=document.source_file,
        row_index=document.row_index,
        document_id=document.id,
    )
