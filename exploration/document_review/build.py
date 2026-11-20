"""Pick 50 Slack threads to read, and the most similar Slack documents to each.

The threads are drawn in three marked groups, by how many messages the parser finds:

- short: at most the 10th percentile of messages per thread;
- long: at least the 90th percentile;
- random: any thread.

For every thread, the 5 Slack documents closest to it by the fine-tuned gte vector (each
Slack document has exactly one vector, so this is one nearest-neighbour query). Threads
are parsed with the LLM-split messages where those exist, like the indexing does. Local,
free; about two minutes to parse all Slack documents.

Writes data/samples.json, read by server.py.

Usage:
    uv run python exploration/document_review/build.py
"""

import json
import random
import statistics
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlmodel import Session

from train_rag.database import get_engine
from train_rag.models import Artifact, SourceType
from train_rag.models.parsed.slack import SlackThread
from train_rag.parsers import ParserRegistry, default_registry
from train_rag.processing.normalize import normalize_content, normalize_text
from train_rag.repository.document import load_artifacts
from train_rag.repository.inferred_messages import load_inferred_messages

DATA = Path(__file__).parent / "data"
OUTPUT = DATA / "samples.json"
DATASET = "enterprise_rag_bench"
TABLE = "slack_embeddings_768_ft"
SEED = 16
GROUPS = {"short": 17, "long": 17, "random": 16}
NEIGHBOURS = 5
NEAREST = text(
    f"select document_id, 1 - (embedding <=> cast(:vector as vector)) as similarity "  # noqa: S608
    f"from {TABLE} where document_id <> :document_id "
    "order by embedding <=> cast(:vector as vector) limit :limit"
)
VECTOR = text(f"select embedding from {TABLE} where document_id = :document_id")  # noqa: S608


def message_count(registry: ParserRegistry, artifact: Artifact) -> int:
    """Number of messages the parser finds in a thread (0 for free text)."""
    parsed = registry.parse(artifact)

    return len(parsed.messages) if isinstance(parsed, SlackThread) else 0


def describe(registry: ParserRegistry, artifact: Artifact) -> dict[str, Any]:
    """A thread as the page shows it: channel, messages and the full cleaned text.

    Args:
        registry: The parsers.
        artifact: The raw Slack document.

    Returns:
        Its id, title, channel, participants, messages, size and full text.
    """
    parsed = registry.parse(artifact)
    full_text = normalize_content(artifact.content)
    thread = parsed if isinstance(parsed, SlackThread) else None

    return {
        "doc_id": artifact.id,
        "document_id": artifact.document_id,
        "title": normalize_text(artifact.title).split("\n")[0][:140],
        "channel": thread.channel if thread else None,
        "participants": thread.participants if thread else [],
        "messages": [
            {"speaker": message.speaker, "role": message.role, "text": normalize_text(message.text)}
            for message in (thread.messages if thread else [])
        ],
        "chars": len(full_text),
        "text": full_text,
    }


def pick(counts: dict[int, int]) -> list[tuple[str, int]]:
    """Draw the marked groups of threads.

    Args:
        counts: Document id -> messages in the thread.

    Returns:
        (group, document id) pairs, short first, then long, then random.
    """
    deciles = statistics.quantiles(counts.values(), n=10)
    short_limit, long_limit = deciles[0], deciles[-1]
    print(f"messages per thread: 10th percentile {short_limit}, 90th {long_limit}")
    pools = {
        "short": sorted(i for i, n in counts.items() if 0 < n <= short_limit),
        "long": sorted(i for i, n in counts.items() if n >= long_limit),
        "random": sorted(counts),
    }
    rng = random.Random(SEED)
    picked: list[tuple[str, int]] = []
    taken: set[int] = set()
    for group, size in GROUPS.items():
        choices = rng.sample([i for i in pools[group] if i not in taken], size)
        taken.update(choices)
        picked += [(group, document_id) for document_id in choices]

    return picked


def neighbours(session: Session, document_id: int) -> list[tuple[int, float]]:
    """The Slack documents closest to one, by the fine-tuned gte vector.

    Args:
        session: Database session.
        document_id: The thread's row id.

    Returns:
        (document id, cosine similarity) pairs, closest first.
    """
    vector = session.execute(VECTOR, {"document_id": document_id}).scalar_one()
    rows = session.execute(
        NEAREST,
        {"vector": str(vector), "document_id": document_id, "limit": NEIGHBOURS},
    ).all()

    return [(row.document_id, round(float(row.similarity), 3)) for row in rows]


def main() -> None:
    """Parse every Slack thread, draw the groups, find neighbours and write the samples."""
    with Session(get_engine()) as session:
        registry = default_registry(inferred_messages=load_inferred_messages(session))
        artifacts = {
            artifact.document_id: artifact
            for artifact in load_artifacts(session, DATASET, SourceType.SLACK)
            if artifact.document_id is not None
        }
        print(f"{len(artifacts):,} Slack documents")
        counts = {i: message_count(registry, artifact) for i, artifact in artifacts.items()}
        samples = []
        for number, (group, document_id) in enumerate(pick(counts), 1):
            similar = neighbours(session, document_id)
            samples.append(
                {
                    "number": number,
                    "group": group,
                    **describe(registry, artifacts[document_id]),
                    "similar": [
                        {**describe(registry, artifacts[other]), "similarity": similarity}
                        for other, similarity in similar
                    ],
                }
            )
    DATA.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(samples, indent=1, ensure_ascii=False))
    print(f"wrote {len(samples)} threads to {OUTPUT}")


if __name__ == "__main__":
    main()
