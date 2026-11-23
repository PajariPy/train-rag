"""Pick the Slack threads to label and extract, spread over channels and lengths.

Threads behind a benchmark question or a dev question are left out, so the later retrieval
test stays clean. The rest are dealt channel by channel and length band by length band, in
turns, so the first 13 (the trial) and the first 200 are both spread out, and the 200 extend
the 13.

Each thread is written as one line per message, "speaker (role): text", the exact text that
DeepSeek and both GLiNER models read. Local and free; about two minutes to parse all Slack
documents.

Writes data/threads.json.

Usage:
    uv run python exploration/slack_extraction/sample.py [--size 200]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sqlmodel import Session

from train_rag.database import get_engine
from train_rag.models import Artifact, SourceType
from train_rag.models.parsed.slack import SlackThread
from train_rag.parsers import ParserRegistry, default_registry
from train_rag.processing.normalize import normalize_content, normalize_text
from train_rag.repository.document import load_artifacts
from train_rag.repository.inferred_messages import load_inferred_messages

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "data"
OUTPUT = DATA / "threads.json"
DATASET = "enterprise_rag_bench"
BENCHMARK = ROOT / "data" / "enterprise_rag_bench" / "data" / "questions" / "test.parquet"
SOURCES = ROOT / "exploration" / "training_questions" / "data" / "sources.json"
SEED = 16
BANDS = ("short", "medium", "long")


def thread_text(registry: ParserRegistry, artifact: Artifact) -> tuple[str, str | None, int]:
    """A thread as the models read it: one "speaker (role): text" line per message.

    Args:
        registry: The parsers.
        artifact: The raw Slack document.

    Returns:
        The text, the channel, and the number of messages (0 when it doesn't parse).
    """
    parsed = registry.parse(artifact)
    if not isinstance(parsed, SlackThread) or not parsed.messages:
        return normalize_content(artifact.content), None, 0
    lines = []
    for message in parsed.messages:
        speaker = f"{message.speaker} ({message.role})" if message.role else message.speaker
        lines.append(f"{speaker}: {normalize_text(message.text)}")

    return "\n".join(lines), parsed.channel, len(parsed.messages)


def held_out() -> set[str]:
    """Doc ids behind a benchmark question or a dev question."""
    benchmark = pd.read_parquet(BENCHMARK)
    ids = {doc_id for doc_ids in benchmark.expected_doc_ids for doc_id in doc_ids}
    ids |= {row["doc_id"] for row in json.loads(SOURCES.read_text()) if row["split"] == "dev"}

    return ids


def deal(threads: list[dict[str, object]], size: int) -> list[dict[str, object]]:
    """Take threads channel by channel and band by band, in turns.

    Args:
        threads: Candidate threads with their channel and length band.
        size: How many to take.

    Returns:
        The picked threads, in dealing order.
    """
    rng = random.Random(SEED)
    pools: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for thread in threads:
        pools[(str(thread["channel"]), str(thread["band"]))].append(thread)
    for pool in pools.values():
        rng.shuffle(pool)
    channels = sorted({channel for channel, _ in pools})
    rng.shuffle(channels)
    picked: list[dict[str, object]] = []
    turn = 0
    while len(picked) < size and any(pools.values()):
        for position, channel in enumerate(channels):
            band = BANDS[(turn + position) % len(BANDS)]
            if pools.get((channel, band)) and len(picked) < size:
                picked.append(pools[(channel, band)].pop())
        turn += 1

    return picked


def main() -> None:
    """Parse the Slack threads, drop the held-out ones and deal the sample."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()

    skip = held_out()
    with Session(get_engine()) as session:
        registry = default_registry(inferred_messages=load_inferred_messages(session))
        artifacts = [
            artifact
            for artifact in load_artifacts(session, DATASET, SourceType.SLACK)
            if artifact.id not in skip
        ]
    print(f"{len(artifacts):,} Slack threads outside the benchmark and dev sets", flush=True)
    threads = []
    for artifact in artifacts:
        text, channel, messages = thread_text(registry, artifact)
        if messages:
            threads.append(
                {"doc_id": artifact.id, "channel": channel, "messages": messages,
                 "words": len(text.split()), "text": text}
            )
    words = sorted(int(thread["words"]) for thread in threads)
    low, high = words[len(words) // 3], words[2 * len(words) // 3]
    for thread in threads:
        size = int(thread["words"])
        thread["band"] = "short" if size < low else "long" if size >= high else "medium"
    picked = deal(threads, args.size)
    DATA.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(picked, indent=1, ensure_ascii=False))
    channels = len({thread["channel"] for thread in picked})
    print(f"wrote {len(picked)} threads from {channels} channels to {OUTPUT}")


if __name__ == "__main__":
    main()
