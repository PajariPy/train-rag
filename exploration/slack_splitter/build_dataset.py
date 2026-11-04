"""Build the test set for the Slack splitter from threads whose messages are known.

Clean threads are taken from the data: every line starts with a speaker, so the parser
splits them correctly and the true messages are known. Each test case changes one thread
the way the merged threads in the data look, or adds text that must not be split:

- 15 `inline`: 1 to 3 messages joined to the one before by a space.
- 10 `escaped`: 1 to 3 messages joined by an escaped line break (backslash-n).
- 10 `timestamp`: every message after the first starts with `(10:07) `.
- 15 cases that must stay as they are: an action-item list naming owners
  (`1) maya: send the deck`), labels (`Next steps: ...`) or config (`max_tokens: 512`)
  added inside a message.

Run: uv run python exploration/slack_splitter/build_dataset.py
"""

import json
import random
import re
from pathlib import Path
from typing import Any

from sqlmodel import Session

from train_rag.database import get_engine
from train_rag.models import SourceType
from train_rag.models.parsed import SlackThread
from train_rag.parsers import SlackParser
from train_rag.repository.document import load_artifacts

DATASET = "enterprise_rag_bench"
DATA_DIR = Path(__file__).parent / "data"
DATASET_FILE = DATA_DIR / "dataset.jsonl"
SEED = 7
HIDDEN_SPEAKER = re.compile(r"(?:[.!?)]\s+|\\n)[a-z][\w.-]*(?: [a-z]+)?: ")
KINDS = {"inline": 15, "escaped": 10, "timestamp": 10, "action_items": 5, "labels": 5, "config": 5}
TASKS = ["send the updated deck", "check the canary logs", "draft the RCA", "ping procurement"]
LABEL_TEXT = " Next steps: confirm the rollback window. Root cause: stale DNS cache on the edge."
CONFIG_TEXT = "\nmax_tokens: 512\ntemperature: 0.2\nname: api-gateway"


def clean_threads(session: Session) -> list[tuple[str, list[tuple[str, str]]]]:
    """Find threads the parser splits cleanly, to serve as truth.

    Kept: 4 to 20 messages, at most 4,000 characters, no roles, and no message with a
    mid-sentence `handle: ` that could be a speaker the parser missed.

    Args:
        session: Database session.

    Returns:
        Title and (speaker, text) messages of each clean thread.
    """
    parser = SlackParser()
    threads = []
    for artifact in load_artifacts(session, DATASET, SourceType.SLACK):
        if len(artifact.content) > 4000:
            continue
        parsed = parser.parse(artifact)
        if not isinstance(parsed, SlackThread) or not 4 <= len(parsed.messages) <= 20:
            continue
        if any(message.role or HIDDEN_SPEAKER.search(message.text) for message in parsed.messages):
            continue
        threads.append(
            (artifact.title, [(message.speaker or "", message.text) for message in parsed.messages])
        )

    return threads


def build_case(
    kind: str, title: str, messages: list[tuple[str, str]], rng: random.Random
) -> dict[str, Any]:
    """Change one clean thread into a test case.

    Args:
        kind: Kind of case, a key of `KINDS`.
        title: Thread title.
        messages: The thread's (speaker, text) messages.
        rng: Random source.

    Returns:
        The case with its content and true messages.
    """
    truth = [list(message) for message in messages]
    lines = [f"{speaker}: {text}" for speaker, text in messages]
    if kind in ("inline", "escaped"):
        joined = set(rng.sample(range(1, len(lines)), k=min(3, rng.randint(1, len(lines) - 1))))
        separator = " " if kind == "inline" else "\\n"
        content = lines[0]
        for index, line in enumerate(lines[1:], start=1):
            content += (separator if index in joined else "\n") + line
    elif kind == "timestamp":
        content = "\n".join(
            [lines[0], *(f"(10:{index * 3:02d}) {line}" for index, line in enumerate(lines[1:], 1))]
        )
    else:
        target = rng.randrange(len(lines))
        names = sorted({speaker for speaker, _text in messages})
        if kind == "action_items":
            owners = rng.sample(names, k=min(2, len(names)))
            extra = " Action items: " + " ".join(
                f"{number}) {owner}: {rng.choice(TASKS)}" for number, owner in enumerate(owners, 1)
            )
        else:
            extra = LABEL_TEXT if kind == "labels" else CONFIG_TEXT
        truth[target][1] += extra
        lines[target] += extra
        content = "\n".join(lines)

    return {"kind": kind, "title": title, "content": content, "messages": truth}


def main() -> None:
    """Write the test cases to `data/dataset.jsonl`."""
    rng = random.Random(SEED)
    with Session(get_engine()) as session:
        threads = clean_threads(session)
    picked = rng.sample(threads, k=sum(KINDS.values()))
    kinds = [kind for kind, count in KINDS.items() for _ in range(count)]
    cases = [
        {"case_id": f"case_{index:02d}", **build_case(kind, title, messages, rng)}
        for index, (kind, (title, messages)) in enumerate(zip(kinds, picked, strict=True), 1)
    ]
    DATA_DIR.mkdir(exist_ok=True)
    DATASET_FILE.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases))
    print(f"Wrote {len(cases)} cases from {len(threads):,} clean threads to {DATASET_FILE}.")


if __name__ == "__main__":
    main()
