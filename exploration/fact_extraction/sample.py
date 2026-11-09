r"""Pick the documents to label for the fact-extraction training set, and the pilot set.

Training set: SIZE documents from every source and every size. Each source is cut into
BANDS size bands of its own (quintiles of its character counts, so a "long" Slack thread and
a "long" meeting transcript are both long for their kind), each source gets at least
FLOOR documents and the rest is shared in proportion to the source's size, and every band of
a source gets an equal share. Empty documents are left out.

Pilot set: the documents of the failure pools (gte's top 50 plus the expected document for
each of the 46 problem cases), kept out of the training set so the model is never trained on
the documents it is tested on.

Usage (from the main checkout, which has the data):
    uv run --project ~/projects/rag-implementation/train-rag python \\
        exploration/fact_extraction/sample.py POOLS.json
"""

import json
import random
import sys
from pathlib import Path

from sqlalchemy import text

from train_rag.database import get_engine

HERE = Path(__file__).parent
OUT = HERE / "data" / "sample.json"
SIZE = 3_000
FLOOR = 150
BANDS = 5
SEED = 7


def allocate(counts: dict[str, int]) -> dict[str, int]:
    """Documents per source: FLOOR each, the rest in proportion to the source's size.

    Args:
        counts: Documents per source in the corpus.

    Returns:
        Documents to sample per source, adding up to SIZE.
    """
    rest = SIZE - FLOOR * len(counts)
    total = sum(counts.values())
    shares = {source: FLOOR + rest * count // total for source, count in counts.items()}
    biggest = max(counts, key=counts.__getitem__)
    shares[biggest] += SIZE - sum(shares.values())

    return shares


def main() -> None:
    """Sample the training set by source and size band, and list the pilot set."""
    pools = json.loads(Path(sys.argv[1]).read_text())["pools"]
    pilot = sorted({doc_id for pool in pools.values() for doc_id in pool["members"]})
    excluded = set(pilot)
    with get_engine().connect() as connection:
        rows = connection.execute(
            text(
                "select doc_id, source_type, char_count, "
                f"ntile({BANDS}) over (partition by source_type order by char_count) "
                "from documents where char_count > 0"
            )
        ).all()

    cells: dict[tuple[str, int], list[str]] = {}
    counts: dict[str, int] = {}
    for doc_id, source, _chars, band in rows:
        counts[source] = counts.get(source, 0) + 1
        if doc_id not in excluded:
            cells.setdefault((source, band), []).append(doc_id)

    generator = random.Random(SEED)
    train: list[dict[str, object]] = []
    plan = []
    for source, share in sorted(allocate(counts).items(), key=lambda item: -item[1]):
        per_band = [share // BANDS + (1 if band < share % BANDS else 0) for band in range(BANDS)]
        for band, wanted in enumerate(per_band, 1):
            picked = generator.sample(sorted(cells[(source, band)]), wanted)
            train.extend({"doc_id": d, "source": source, "band": band} for d in picked)
        plan.append(
            {"source": source, "corpus": counts[source], "sample": share, "per_band": per_band}
        )

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"train": train, "pilot": pilot, "plan": plan}, indent=1))
    for row in plan:
        print(
            f"{row['source']:13} {row['corpus']:>8,} -> {row['sample']:>5}  bands {row['per_band']}"
        )
    print(f"train {len(train)}, pilot {len(pilot)} (kept out of train); wrote {OUT}")


main()
