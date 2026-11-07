"""Embed the summaries the documents already contain and index them for search."""

import time

from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import create_tables, get_engine
from train_rag.embedder import Embedder, MiniLmEmbedder
from train_rag.models import SourceType
from train_rag.parsers import default_registry
from train_rag.repository.document import load_artifacts
from train_rag.repository.embedding import create_vector_index, replace_embeddings
from train_rag.tables import EMBEDDING_TABLES

FIELD = "summary"
FREE_SUMMARIES: dict[SourceType, str] = {
    SourceType.FIREFLIES: "summary",
    SourceType.GITHUB: "description",
    SourceType.JIRA: "issue_summary",
}


def index_summaries(embedder: Embedder | None = None) -> None:
    """Embed every free summary into its source's table and build the vector index.

    Free summaries are the parsed fields that already summarize a document: the
    Fireflies summary, the GitHub description and the Jira issue summary. Documents kept
    as open text have none and are skipped. All are stored under the field `summary`.

    Args:
        embedder: Embedding model; MiniLM when None.
    """
    embedder = embedder or MiniLmEmbedder()
    engine = get_engine()
    create_tables(engine)
    registry = default_registry()

    with Session(engine) as session:
        for source_type, field in FREE_SUMMARIES.items():
            started = time.perf_counter()
            parser = registry.get(source_type)
            if parser is None:
                continue

            summaries: list[tuple[int, str, str]] = []
            for artifact in load_artifacts(session, EnterpriseRagBenchLoader.name, source_type):
                summary = getattr(parser.parse(artifact), field, None)
                if summary and artifact.document_id is not None:
                    summaries.append((artifact.document_id, artifact.id, summary))

            vectors = embedder.embed([summary for _document_id, _doc_id, summary in summaries])
            table = EMBEDDING_TABLES[embedder.dimensions][source_type]
            records = [
                table(
                    document_id=document_id,
                    doc_id=doc_id,
                    field=FIELD,
                    model=embedder.name,
                    text=summary,
                    embedding=vector,
                )
                for (document_id, doc_id, summary), vector in zip(summaries, vectors, strict=True)
            ]
            replace_embeddings(session, table, embedder.name, FIELD, records)
            create_vector_index(session, table)
            print(
                f"{source_type}: embedded {len(records):,} {field} texts into "
                f"{table.__tablename__} in {time.perf_counter() - started:.0f}s"
            )
