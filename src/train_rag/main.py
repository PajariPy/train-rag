"""Command-line starting point: `uv run train-rag <command>`."""

import argparse
import asyncio
from pathlib import Path

from train_rag.download import DEFAULT_DATA_DIR, download_dataset
from train_rag.embedder import EMBEDDERS
from train_rag.evaluate import evaluate, evaluate_search
from train_rag.evaluation.registry import JUDGES, RETRIEVERS, SYSTEMS
from train_rag.import_db import import_corpus
from train_rag.index_bm25 import index_bm25
from train_rag.index_summaries import index_summaries
from train_rag.models import QuestionType
from train_rag.pipelines.data_ingestion import run_data_ingestion
from train_rag.pipelines.retrieval import run_retrieval
from train_rag.progress.page import serve_progress_page


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser with one subcommand per pipeline step.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(prog="train-rag", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="download EnterpriseRAG-Bench from Hugging Face")
    commands.add_parser("import-db", help="import the downloaded dataset into Postgres")
    commands.add_parser("setup", help="download the dataset, then import it")
    commands.add_parser("index-summaries", help="embed the documents' own summaries for search")
    pipeline = commands.add_parser(
        "ingest", help="load, parse, normalize, chunk, embed and store every document"
    )
    pipeline.add_argument("--embedder", choices=sorted(EMBEDDERS), default="minilm")
    pipeline.add_argument("--fresh", action="store_true", help="delete stored chunks first")
    pipeline.add_argument(
        "--redo-lists",
        action="store_true",
        help="re-embed the documents stored as a list literal, now decoded to plain text",
    )
    pipeline.add_argument(
        "--sample", type=int, help="only the cited documents plus N random distractors"
    )
    page = commands.add_parser("progress-page", help="serve a web page with the run's progress")
    page.add_argument("--port", type=int, default=8765)
    keywords = commands.add_parser("index-bm25", help="build a BM25 keyword index on disk")
    keywords.add_argument(
        "--sample", type=int, help="only the cited documents plus N random distractors"
    )
    retrieve = commands.add_parser(
        "retrieve", help="find documents for a query (BM25 + gte, reranked)"
    )
    retrieve.add_argument("query", help="the question or search text")
    retrieve.add_argument("--limit", type=int, default=10, help="number of documents")
    retrieve.add_argument("--no-rerank", action="store_true", help="keep the fusion's order")
    search = commands.add_parser("evaluate-search", help="measure retrieval recall, no LLM")
    search.add_argument("--retriever", choices=sorted(RETRIEVERS), required=True)
    search.add_argument("--sample", type=int, help="sample size the BM25 index was built for")
    search.add_argument("--question-type", type=QuestionType, choices=list(QuestionType))
    search.add_argument("--limit", type=int, help="evaluate only the first N questions")
    scoring = commands.add_parser("evaluate", help="score a RAG system on the benchmark")
    scoring.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    scoring.add_argument("--judge", choices=sorted(JUDGES), default="gemini")
    scoring.add_argument("--question-type", type=QuestionType, choices=list(QuestionType))
    scoring.add_argument("--limit", type=int, help="evaluate only the first N questions")
    scoring.add_argument("--workers", type=int, default=8)
    scoring.add_argument(
        "--wrong-in", type=int, metavar="RUN", help="only the questions this run got wrong"
    )

    return parser


def main() -> None:
    """Run the requested pipeline step."""
    args = build_parser().parse_args()

    if args.command in ("download", "setup"):
        path = download_dataset(args.data_dir)
        print(f"EnterpriseRAG-Bench downloaded to {path}")

    if args.command in ("import-db", "setup"):
        import_corpus(args.data_dir)

    if args.command == "index-summaries":
        index_summaries()

    if args.command == "ingest":
        run_data_ingestion(EMBEDDERS[args.embedder](), args.sample, args.fresh, args.redo_lists)

    if args.command == "progress-page":
        serve_progress_page(args.port)

    if args.command == "index-bm25":
        index_bm25(args.sample)

    if args.command == "retrieve":
        documents = asyncio.run(run_retrieval(args.query, args.limit, not args.no_rerank))
        for rank, document in enumerate(documents, 1):
            ranks = ", ".join(f"{name} #{place}" for name, place in document.ranks.items())
            score = document.rerank_score
            rerank = "" if score is None else f"rerank {score:.3f} · "
            title = document.title.strip().split("\n")[0][:80]  # some titles hold a whole page
            print(f"{rank:>2}. {title} [{document.source_type.value}]")
            print(f"    {document.doc_id} · {rerank}rrf {document.score:.4f} · {ranks}")

    if args.command == "evaluate-search":
        evaluate_search(args.retriever, args.question_type, args.limit, args.sample)

    if args.command == "evaluate":
        evaluate(
            args.system,
            args.judge,
            args.question_type,
            args.limit,
            args.workers,
            args.wrong_in,
        )


if __name__ == "__main__":
    main()
