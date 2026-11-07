"""Data ingestion: load, parse, normalize, chunk, embed and store every document.

Documents go through the steps in batches, and each batch is committed in one
transaction: its parsed fields to `parsed_documents`, its chunks and vectors to the
source's embedding table. A document with chunks from the model is done, so on start
the pipeline plans only the documents left. Stop it at any time and run it again to
resume. Progress is recorded for the page served by `train-rag progress-page`.
"""

from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import create_tables, get_engine
from train_rag.embedder import Embedder, GteModernBertEmbedder, MiniLmEmbedder
from train_rag.models import SourceType
from train_rag.parsers import default_registry
from train_rag.processing.chunking import RecursiveChunker, chunk_documents
from train_rag.processing.normalize import decode_string_list, normalize_documents
from train_rag.processing.parsing import parse_documents
from train_rag.progress.recorder import RunProgress
from train_rag.repository.document import bracketed_documents, load_documents
from train_rag.repository.embedding import (
    create_vector_index,
    delete_document_embeddings,
    delete_embeddings,
    drop_vector_index,
    pending_documents,
    store_chunks,
)
from train_rag.repository.inferred_attendees import load_inferred_attendees
from train_rag.repository.inferred_date import load_inferred_dates
from train_rag.repository.inferred_messages import load_inferred_messages
from train_rag.repository.parsed_document import upsert_parsed_documents
from train_rag.sampling import sample_documents
from train_rag.tables import EMBEDDING_TABLES

FIELD = "chunk"
DOCUMENT_BATCH = 500

# Chunk sizes per model, in characters (about 3.7 characters per token). MiniLM reads
# 256 word pieces: 600 characters plus 100 of context on each side and the title fit in
# 95% of chunks. gte-modernbert reads 8,192 tokens, so most documents stay whole.
CHUNKERS: dict[str, RecursiveChunker] = {
    MiniLmEmbedder.name: RecursiveChunker(size=600, context=100),
    GteModernBertEmbedder.name: RecursiveChunker(size=24_000, context=500),
}


def forget_list_documents(session: Session, dataset: str, embedder: Embedder) -> None:
    """Delete this model's chunks of the documents stored as a list literal.

    Their chunks were made before normalization decoded such lists into plain text
    (`decode_string_list`), so they were embedded as escaped list syntax. Without chunks
    they count as pending again, and the run re-embeds them from the decoded text.

    Args:
        session: Active database session.
        dataset: Dataset to fix.
        embedder: Embedding model whose chunks are deleted.
    """
    found: dict[SourceType, list[int]] = {}
    for document_id, source_type, content in bracketed_documents(session, dataset):
        if decode_string_list(content) != content:
            found.setdefault(source_type, []).append(document_id)
    tables = EMBEDDING_TABLES[embedder.dimensions]
    for source_type, document_ids in found.items():
        deleted = delete_document_embeddings(
            session, tables[source_type], embedder.name, FIELD, document_ids
        )
        print(
            f"{source_type.value}: {len(document_ids):,} list documents, {deleted:,} chunks deleted"
        )


def run_data_ingestion(
    embedder: Embedder, sample: int | None = None, fresh: bool = False, redo_lists: bool = False
) -> None:
    """Ingest every document that has no chunks from this embedding model yet.

    Args:
        embedder: Embedding model; its name picks the chunk size from `CHUNKERS`.
        sample: Only the test sample (every cited document plus this many random
            distractors), or None for the whole corpus.
        fresh: Delete this model's chunks first and start over.
        redo_lists: Re-embed the documents stored as a list literal (see
            `forget_list_documents`).
    """
    engine = get_engine()
    create_tables(engine)
    dataset = EnterpriseRagBenchLoader.name
    chunker = CHUNKERS[embedder.name]
    tables = EMBEDDING_TABLES[embedder.dimensions]

    with Session(engine) as session:
        # Plan: what is left to do.
        if fresh:
            for table in tables.values():
                delete_embeddings(session, table, embedder.name, FIELD)
        elif redo_lists:
            forget_list_documents(session, dataset, embedder)
        selected = sample_documents(session, dataset, sample) if sample is not None else None
        plan = {
            source_type: pending_documents(
                session, dataset, source_type, tables[source_type], embedder.name, FIELD, selected
            )
            for source_type in SourceType
        }
        registry = default_registry(
            load_inferred_dates(session),
            load_inferred_attendees(session),
            load_inferred_messages(session),
        )
        total_documents = sum(len(pending) for pending in plan.values())
        total_chars = sum(chars for pending in plan.values() for _id, chars in pending)
        name = f"data-ingestion {embedder.name}" + (f" (sample {sample:,})" if sample else "")
        print(f"{name}: {total_documents:,} documents to process ({total_chars / 1e6:,.0f}M chars)")

        with RunProgress(engine, name, total_documents, total_chars, "documents") as progress:
            for source_type, pending in plan.items():
                if not pending:
                    continue
                table = tables[source_type]
                drop_vector_index(session, table)
                for start in range(0, len(pending), DOCUMENT_BATCH):
                    batch = pending[start : start + DOCUMENT_BATCH]

                    # 1. Load
                    documents = load_documents(session, [document_id for document_id, _ in batch])
                    # 2. Parse
                    upsert_parsed_documents(session, parse_documents(registry, documents))
                    # 3. Normalize
                    clean = normalize_documents(documents)
                    # 4. Chunk
                    chunks = chunk_documents(chunker, clean)
                    # 5. Embed
                    vectors = embedder.embed([chunk.text for chunk in chunks])
                    # 6. Store
                    store_chunks(session, table, embedder.name, FIELD, chunks, vectors)
                    session.commit()

                    progress.advance(
                        source_type.value, len(batch), sum(chars for _id, chars in batch)
                    )
                progress.advance(f"{source_type.value}: building the vector index", 0, 0)
                create_vector_index(session, table)
