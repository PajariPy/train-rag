"""Build a labelled set of 50 Slack threads for evaluating the channel judge.

Threads come from Postgres and are ones whose title is their channel, so the true
channel is known. Every title is replaced with a file-name title, like the real
threads that lost their channel. 25 threads get their own channel hidden in the text
in one of five ways. The other 25 do not reveal their channel: 15 never mention one,
and 10 mention a different channel (a pin, a digest, a cross-post), which must still
count as no channel.

Run: uv run python exploration/channel_judge/build_dataset.py
"""

import json
import random
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from sqlmodel import Session

from train_rag.database import get_engine
from train_rag.models import Artifact, SourceType
from train_rag.parsers import SlackParser
from train_rag.repository.document import load_artifacts

DATASET = "enterprise_rag_bench"
DATASET_FILE = Path(__file__).parent / "data" / "dataset.jsonl"
POSITIVES = 25
NEGATIVES = 25
DISTRACTED_NEGATIVES = 10
POOL_SIZE = 20_000
MIN_CHANNEL_THREADS = 20
SEED = 7
MIN_CHARS = 800
MAX_CHARS = 5000
CHANNEL_MENTION = re.compile(r"(?<![\w&])#[a-z][\w-]*")

Placement = Callable[[str, str, random.Random], str]


def header_line(content: str, channel: str, rng: random.Random) -> str:
    """Put a `Channel: #name` line above the thread.

    Args:
        content: Thread text.
        channel: Channel to hide.
        rng: Random source, unused here.

    Returns:
        The thread with the channel in a header line.
    """
    return f"Channel: #{channel}\n\n{content}"


def export_header(content: str, channel: str, rng: random.Random) -> str:
    """Put an export banner without a `#` above the thread.

    Args:
        content: Thread text.
        channel: Channel to hide.
        rng: Random source, unused here.

    Returns:
        The thread with the channel in an export banner.
    """
    return f"Slack export · {channel} · thread (UTC)\n\n{content}"


def hashtag_in_message(content: str, channel: str, rng: random.Random) -> str:
    """Append a `#name` remark to a message in the middle of the thread.

    Args:
        content: Thread text.
        channel: Channel to hide.
        rng: Random source used to pick the message.

    Returns:
        The thread with one message mentioning the channel.
    """
    lines = content.splitlines()
    starts = [index for index, line in enumerate(lines) if ": " in line[:40]]
    index = starts[len(starts) // 2] if starts else len(lines) // 2
    remark = rng.choice(
        [
            f" (keeping this in #{channel} so the right folks see it)",
            f" — let's keep the follow-ups here in #{channel}.",
            f" (posting here in #{channel} since it's our team channel)",
        ]
    )
    lines[index] = lines[index] + remark

    return "\n".join(lines)


def plain_words_in_message(content: str, channel: str, rng: random.Random) -> str:
    """Append a mention without `#` to the first message.

    Args:
        content: Thread text.
        channel: Channel to hide.
        rng: Random source used to pick the wording.

    Returns:
        The thread whose first message names the channel in plain words.
    """
    lines = content.splitlines()
    remark = rng.choice(
        [
            f" Posting in the {channel} channel since this is their area.",
            f" Dropping it in {channel} instead of DMs so it's searchable.",
        ]
    )
    lines[0] = lines[0] + remark

    return "\n".join(lines)


def bot_footer(content: str, channel: str, rng: random.Random) -> str:
    """Add a bot message at the end that names the channel.

    Args:
        content: Thread text.
        channel: Channel to hide.
        rng: Random source used to pick the bot line.

    Returns:
        The thread with a closing bot message naming the channel.
    """
    footer = rng.choice(
        [
            f"slackbot: This thread in #{channel} was marked resolved.",
            f"slackbot: Reminder for #{channel}: this thread has been open for 3 days.",
        ]
    )

    return f"{content}\n{footer}"


def other_channel_mention(content: str, other: str, rng: random.Random) -> str:
    """Mention a different channel than the one the thread was posted in.

    Args:
        content: Thread text.
        other: A channel the thread was not posted in.
        rng: Random source used to pick the wording and the message.

    Returns:
        The thread with a remark that points to the other channel.
    """
    lines = content.splitlines()
    remark = rng.choice(
        [
            f" fyi I pinned this in #{other}.",
            f" ask in #{other} if you need access.",
            f" cross-posting to #{other} for visibility.",
            f" let's move the follow-up to #{other}.",
        ]
    )
    if rng.random() < 0.3:
        return f"{content}\ntriage-bot: archived to the {other} digest for this week."

    index = rng.randrange(len(lines))
    lines[index] = lines[index] + remark

    return "\n".join(lines)


PLACEMENTS: dict[str, Placement] = {
    "header_line": header_line,
    "export_header": export_header,
    "hashtag_in_message": hashtag_in_message,
    "plain_words_in_message": plain_words_in_message,
    "bot_footer": bot_footer,
}


def file_name_title(artifact: Artifact, rng: random.Random) -> str:
    """Make a title like the real channel-less threads: `<digits>-<slug>.json`.

    Args:
        artifact: Thread whose first message gives the slug.
        rng: Random source for the numeric prefix.

    Returns:
        A file-name title that does not contain the channel.
    """
    first_words = re.findall(r"[a-z0-9]+", artifact.content.lower())[2:7]

    return f"{rng.randint(1_700_000_000, 1_900_000_000)}-{'-'.join(first_words)}.json"


def mentions_channel(text: str, channel: str) -> bool:
    """Tell whether text already mentions a channel, so the label would be unclear.

    Args:
        text: Thread text.
        channel: The thread's real channel.

    Returns:
        True when the text has any `#name` or the channel name as a word.
    """
    own_name = re.search(rf"(?<![\w-]){re.escape(channel)}(?![\w-])", text, re.IGNORECASE)

    return bool(CHANNEL_MENTION.search(text) or own_name)


def candidates(session: Session, rng: random.Random) -> list[tuple[Artifact, str]]:
    """Pick clean threads posted in a real channel that never mention a channel.

    The parser also takes numbers, sentences and one-off slugs from the title as a
    channel, so only names that recur in at least `MIN_CHANNEL_THREADS` threads count.

    Args:
        session: Database session.
        rng: Random source for the shuffle.

    Returns:
        Threads with their channel, in random order.
    """
    parser = SlackParser()
    threads = [
        (artifact, parser.parse(artifact))
        for artifact in load_artifacts(session, DATASET, SourceType.SLACK, limit=POOL_SIZE)
    ]
    counts = Counter(
        thread.channel for _artifact, thread in threads if thread.formats.get("channel")
    )
    picked: list[tuple[Artifact, str]] = []
    for artifact, thread in threads:
        channel = thread.channel
        if thread.formats.get("channel") != "channel_from_title" or channel is None:
            continue
        if counts[channel] < MIN_CHANNEL_THREADS:
            continue
        if not MIN_CHARS <= len(artifact.content) <= MAX_CHARS or len(thread.messages) < 3:
            continue
        if mentions_channel(artifact.content, channel):
            continue

        picked.append((artifact, channel))
    rng.shuffle(picked)

    return picked


def main() -> None:
    """Build the dataset and write it as JSON lines."""
    rng = random.Random(SEED)
    with Session(get_engine()) as session:
        pool = candidates(session, rng)

    rows = []
    placements = list(PLACEMENTS)
    channels = sorted({channel for _artifact, channel in pool})
    for index, (artifact, channel) in enumerate(pool[: POSITIVES + NEGATIVES]):
        positive = index < POSITIVES
        if positive:
            placement = placements[index % len(placements)]
            content = PLACEMENTS[placement](artifact.content, channel, rng)
        elif index < POSITIVES + DISTRACTED_NEGATIVES:
            placement = "other_channel_mention"
            other = rng.choice([name for name in channels if name != channel])
            content = other_channel_mention(artifact.content, other, rng)
        else:
            placement = "no_mention"
            content = artifact.content
        rows.append(
            {
                "id": f"case_{index + 1:02d}",
                "document_id": artifact.document_id,
                "title": file_name_title(artifact, rng),
                "content": content,
                "has_channel": positive,
                "channel": channel if positive else None,
                "placement": placement,
            }
        )

    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATASET_FILE.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"Wrote {len(rows)} cases to {DATASET_FILE} from {len(pool)} clean threads.")


if __name__ == "__main__":
    main()
