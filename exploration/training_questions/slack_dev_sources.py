"""Pick Slack threads for an extended semantic dev split (`dev_slack`).

The first dev split has only 11 semantic questions with a Slack answer, too few to measure
whether mined Slack fields help search. This adds 200 Slack sources for semantic questions,
written and mined exactly like the other splits (generate.py, then mine.py).

Left out as sources:
- the benchmark's right documents and every existing train or dev source, each with its 20
  nearest documents (so no source is a twin of a question we already have);
- every training negative (the fine-tuned gte was trained to push those away);
- the threads sampled for field extraction (exploration/slack_extraction/data/threads.json),
  which will feed the extractor's training;
- the off-topic folders, and threads too short for a question.

Appends the rows, split "dev_slack", to data/sources.json (a backup is written first).
Free, local; a few minutes for the neighbour searches.

Usage:
    uv run python exploration/training_questions/slack_dev_sources.py
"""

import json
import random
import shutil
import sys
from pathlib import Path

from sqlalchemy import text

from train_rag.database import get_engine
from train_rag.progress.recorder import RunProgress
from train_rag.training.quotas import is_usable_source

sys.path.insert(0, str(Path(__file__).parent))
from sources import UUID_INDEX, is_misc, nearest_documents  # noqa: E402

DATA = Path(__file__).parent / "data"
SOURCES = DATA / "sources.json"
EXCLUDED = DATA / "excluded.json"
TRAINING_SET = DATA / "training_set.jsonl"
EXTRACTION_SAMPLE = Path(__file__).parents[1] / "slack_extraction" / "data" / "threads.json"
SPLIT = "dev_slack"
SIZE = 200
SEED = 17


def main() -> None:
    """Build the exclusion set, draw the Slack sources and append them."""
    sources: list[dict[str, str]] = json.loads(SOURCES.read_text())
    if any(row["split"] == SPLIT for row in sources):
        raise SystemExit(f"{SOURCES} already has {SPLIT} rows")
    paths: dict[str, str] = json.loads(UUID_INDEX.read_text())
    evaluation = json.loads(EXCLUDED.read_text())["evaluation"]
    negatives = {
        doc_id
        for line in TRAINING_SET.read_text().splitlines()
        if line
        for doc_id in json.loads(line)["negatives"]
    }
    extraction = {row["doc_id"] for row in json.loads(EXTRACTION_SAMPLE.read_text())}
    with get_engine().connect() as connection:
        known = sorted(set(evaluation) | {row["doc_id"] for row in sources})
        print(f"neighbours of {len(known):,} known documents", flush=True)
        excluded = set(known) | negatives | extraction
        with RunProgress(
            get_engine(), "dev_slack sources: neighbours of known documents", len(known)
        ) as progress:
            for index, doc_id in enumerate(known, 1):
                excluded.update(nearest_documents(connection, doc_id))
                if index % 50 == 0 or index == len(known):
                    progress.advance("neighbour search", 50 if index % 50 == 0 else index % 50,
                                     note="local, $0")
        slack = {
            str(row.doc_id): int(row.char_count)
            for row in connection.execute(
                text("select doc_id, char_count from documents where source_type = 'slack'")
            ).all()
        }
    pool = sorted(
        doc_id
        for doc_id, chars in slack.items()
        if doc_id not in excluded
        and not is_misc(paths.get(doc_id, ""))
        and is_usable_source(chars, "semantic")
    )
    print(f"{len(excluded):,} excluded; {len(pool):,} Slack threads left to draw from")
    picked = random.Random(SEED).sample(pool, SIZE)
    rows = [
        {"doc_id": doc_id, "source_type": "slack", "question_type": "semantic", "split": SPLIT}
        for doc_id in picked
    ]
    shutil.copy(SOURCES, SOURCES.with_suffix(".json.bak"))
    SOURCES.write_text(json.dumps(sources + rows, indent=1))
    print(f"appended {len(rows)} {SPLIT} sources to {SOURCES} (backup: sources.json.bak)")


if __name__ == "__main__":
    main()
