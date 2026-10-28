"""Command-line starting point: `uv run train-rag <command>`."""

import argparse
from pathlib import Path

from train_rag.download import DEFAULT_DATA_DIR, download_dataset
from train_rag.import_db import import_corpus


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

    return parser


def main() -> None:
    """Run the requested pipeline step."""
    args = build_parser().parse_args()

    if args.command in ("download", "setup"):
        path = download_dataset(args.data_dir)
        print(f"EnterpriseRAG-Bench downloaded to {path}")

    if args.command in ("import-db", "setup"):
        import_corpus(args.data_dir)


if __name__ == "__main__":
    main()
