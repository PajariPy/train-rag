"""Store the batch splitter's Slack messages in Postgres, where the Slack parser reads them.

Only threads whose cut worked and differs from the parser's split are stored (see
`run_batch.py collect`); rerunning replaces this splitter's rows.

Run: uv run python exploration/slack_splitter/store_messages.py
"""

import json
from pathlib import Path

from splitter import MODEL
from sqlmodel import Session

from train_rag.database import create_tables, get_engine
from train_rag.models.parsed import SlackMessage
from train_rag.repository.inferred_messages import replace_inferred_messages

SPLITS_FILE = Path(__file__).parent / "data" / "batch_splits.jsonl"
METHOD = f"{MODEL} slack splitter"


def main() -> None:
    """Read the batch splits and store them."""
    rows = [json.loads(line) for line in SPLITS_FILE.read_text().splitlines()]
    threads = [
        (row["document_id"], [SlackMessage.model_validate(message) for message in row["messages"]])
        for row in rows
    ]
    engine = get_engine()
    create_tables(engine)
    with Session(engine) as session:
        replace_inferred_messages(session, METHOD, threads)

    print(f"Stored {len(threads):,} split threads as {METHOD!r}.")


if __name__ == "__main__":
    main()
