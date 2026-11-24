"""Command-line steps for mined fields: import a labels file, embed and index the profiles."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import create_tables, get_engine
from train_rag.embedder import FineTunedGteEmbedder
from train_rag.mined_fields.embed import embed_field_profiles
from train_rag.mined_fields.importer import import_fields
from train_rag.mined_fields.stored import build_field_search

DEEPSEEK_LABELS = Path("exploration/slack_extraction/data/labels.jsonl")
DEEPSEEK_EXTRACTOR = "deepseek-fc8ae8878988"
DEEPSEEK_PROMPT = "fc8ae8878988"
EMBED_BATCH = 64


def add_field_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add the `import-fields`, `embed-fields` and `index-fields` commands.

    Args:
        commands: The CLI's subcommands.
    """
    importing = commands.add_parser(
        "import-fields", help="store mined fields from a labels file (DeepSeek or the SLM)"
    )
    importing.add_argument("--file", type=Path, default=DEEPSEEK_LABELS)
    importing.add_argument("--extractor", default=DEEPSEEK_EXTRACTOR, help="name of the model")
    importing.add_argument(
        "--prompt-key", default=DEEPSEEK_PROMPT, help="only lines of this prompt hash ('' = all)"
    )
    commands.add_parser(
        "embed-fields", help="embed the mined-field profiles with the fine-tuned gte (local)"
    )
    indexing = commands.add_parser(
        "index-fields", help="build field search's table and DiskANN indexes (after embed-fields)"
    )
    indexing.add_argument("--workflow", default=None, help="workflow key to report the step in")


def run_field_command(args: argparse.Namespace) -> None:
    """Run `import-fields`, `embed-fields` or `index-fields`.

    Args:
        args: Parsed command-line arguments.
    """
    load_dotenv(Path(".env"))
    engine = get_engine()
    create_tables(engine)

    if args.command == "import-fields":
        dataset = EnterpriseRagBenchLoader.name
        stored = import_fields(engine, args.file, args.extractor, args.prompt_key or None, dataset)
        print(f"Stored {stored:,} mined-field rows from {args.file} as {args.extractor}")

    if args.command == "embed-fields":
        done = embed_field_profiles(engine, FineTunedGteEmbedder(batch_size=EMBED_BATCH))
        print(f"Embedded the profiles of {done:,} mined-field rows")

    if args.command == "index-fields":
        model = FineTunedGteEmbedder.name
        indexed = build_field_search(engine, model, args.workflow)
        print(f"Indexed {indexed:,} mined documents for field search ({model})")
