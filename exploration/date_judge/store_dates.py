"""Store the batch judge's meeting dates in Postgres, where the Fireflies parser reads them.

Only meetings the judge found a date for are stored; rerunning replaces this judge's
rows. A clue without a year (`by Tue 9/14`) is stored with `year_in_evidence` False,
since the year may be a guess.

Run: uv run python exploration/date_judge/store_dates.py
"""

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

from judge import MODEL
from sqlmodel import Session

from train_rag.database import create_tables, get_engine
from train_rag.models.parsed import InferredDate
from train_rag.repository.inferred_date import replace_inferred_dates

PREDICTIONS_FILE = Path(__file__).parent / "data" / "batch_predictions.jsonl"
METHOD = f"{MODEL} date judge"
YEAR = re.compile(r"(?<!\d)20[2-3]\d(?!\d)|(?<!\d)20[2-3]\d[01]\d[0-3]\d(?!\d)")


def main() -> None:
    """Read the batch predictions and store the dates found."""
    rows = [json.loads(line) for line in PREDICTIONS_FILE.read_text().splitlines()]
    dates = [
        (
            row["document_id"],
            InferredDate(
                held_on=date.fromisoformat(row["held_on"]),
                precision=row["precision"] or "week",
                evidence=row["evidence"] or "",
                year_in_evidence=bool(YEAR.search(row["evidence"] or "")),
            ),
        )
        for row in rows
        if row["reveals_date"] and row["held_on"]
    ]
    engine = get_engine()
    create_tables(engine)
    with Session(engine) as session:
        replace_inferred_dates(session, METHOD, dates)

    years = Counter(inferred.year_in_evidence for _document_id, inferred in dates)
    print(
        f"Stored {len(dates)} of {len(rows)} meetings as {METHOD!r}: "
        f"{years[True]} with the year in the evidence, {years[False]} without."
    )


if __name__ == "__main__":
    main()
