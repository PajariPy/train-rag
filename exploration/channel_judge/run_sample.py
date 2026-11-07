"""Run the channel judge on a random sample of the Slack threads without a channel.

The sample is large enough to estimate, within about ±3% at 95% confidence, how many of
all channel-less threads reveal their channel in the text. It is stratified by the kind
of title, so each kind gets its own estimate. The run stops before it spends more than
`BUDGET_USD`, using the token counts Gemini reports.

Run: uv run python exploration/channel_judge/run_sample.py
"""

import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from judge import MODEL, judge_thread
from sqlmodel import Session

from train_rag.database import get_engine
from train_rag.models import Artifact, SourceType
from train_rag.parsers import SlackParser
from train_rag.repository.document import load_artifacts

DATASET = "enterprise_rag_bench"
ENV_FILE = Path(__file__).parents[2] / ".env"
RESULTS_FILE = Path(__file__).parent / "data" / "sample_predictions.jsonl"
REPORT_FILE = Path(__file__).parent / "data" / "sample_report.md"
SEED = 11
CONFIDENCE_Z = 1.96
MARGIN = 0.03
BUDGET_USD = 2.00
INPUT_PRICE = 0.75 / 1_000_000
OUTPUT_PRICE = 3.75 / 1_000_000
CHUNK = 80
WORKERS = 8


def sample_size(population: int) -> int:
    """Threads needed to estimate a share within `MARGIN` at 95% confidence.

    Uses the worst case (a 50% share) with the finite population correction.

    Args:
        population: Number of threads the estimate is about.

    Returns:
        The sample size.
    """
    infinite = CONFIDENCE_Z**2 * 0.25 / MARGIN**2

    return math.ceil(infinite / (1 + (infinite - 1) / population))


def title_kind(title: str) -> str:
    """Group a channel-less thread by what its title is instead of a channel.

    Args:
        title: Thread title.

    Returns:
        The kind of title.
    """
    stripped = title.strip()
    if not stripped:
        return "empty"
    if stripped.endswith(".json") or "/" in stripped:
        return "file name"
    if stripped.isdigit():
        return "number"
    if " " in stripped:
        return "sentence or first message"

    return "slug or id"


def channel_less_threads() -> list[Artifact]:
    """Load every Slack thread the parser flags for a missing channel.

    Returns:
        The threads, in database order.
    """
    parser = SlackParser()
    with Session(get_engine()) as session:
        return [
            artifact
            for artifact in load_artifacts(session, DATASET, SourceType.SLACK)
            if "channel" in parser.parse(artifact).missing_fields
        ]


def stratified_sample(threads: list[Artifact], size: int, rng: random.Random) -> list[Artifact]:
    """Draw a sample with each title kind in proportion to its share of all threads.

    Args:
        threads: All channel-less threads.
        size: Total sample size.
        rng: Random source.

    Returns:
        The sampled threads, shuffled.
    """
    by_kind: dict[str, list[Artifact]] = defaultdict(list)
    for artifact in threads:
        by_kind[title_kind(artifact.title)].append(artifact)

    picked: list[Artifact] = []
    for group in by_kind.values():
        count = max(1, round(size * len(group) / len(threads)))
        picked += rng.sample(group, min(count, len(group)))
    rng.shuffle(picked)

    return picked


def wilson_interval(found: int, total: int) -> tuple[float, float]:
    """95% confidence interval for a share, which stays sensible near 0% and 100%.

    Args:
        found: Threads that revealed their channel.
        total: Threads judged.

    Returns:
        The lower and upper bound of the share.
    """
    if total == 0:
        return 0.0, 0.0

    share = found / total
    z2 = CONFIDENCE_Z**2
    centre = (share + z2 / (2 * total)) / (1 + z2 / total)
    spread = (
        CONFIDENCE_Z
        * math.sqrt(share * (1 - share) / total + z2 / (4 * total**2))
        / (1 + z2 / total)
    )

    return max(0.0, centre - spread), min(1.0, centre + spread)


def judge_sample(sample: list[Artifact]) -> list[dict[str, Any]]:
    """Judge the sample in chunks, stopping before the budget runs out.

    Args:
        sample: Threads to judge.

    Returns:
        One result row per judged thread.
    """
    load_dotenv(ENV_FILE)
    client = genai.Client()
    rows: list[dict[str, Any]] = []
    spent = 0.0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(sample), CHUNK):
            per_thread = spent / len(rows) if rows else 0.0
            if spent + per_thread * CHUNK > BUDGET_USD:
                print(f"Stopping at {len(rows)} threads to stay within ${BUDGET_USD:.2f}.")
                break

            chunk = sample[start : start + CHUNK]
            results = pool.map(lambda item: judge_thread(client, item.title, item.content), chunk)
            for artifact, result in zip(chunk, results, strict=True):
                spent += result.input_tokens * INPUT_PRICE + result.output_tokens * OUTPUT_PRICE
                verdict = result.verdict
                rows.append(
                    {
                        "document_id": artifact.document_id,
                        "title": artifact.title[:200],
                        "title_kind": title_kind(artifact.title),
                        "contains_channel": verdict.contains_channel if verdict else None,
                        "channel": verdict.channel if verdict else None,
                        "evidence": verdict.evidence if verdict else None,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    }
                )
            print(f"{len(rows)}/{len(sample)} judged, ${spent:.2f} spent")

    return rows


def report(rows: list[dict[str, Any]], population: Counter[str]) -> str:
    """Estimate how many channel-less threads reveal their channel, overall and per kind.

    Args:
        rows: Judged sample.
        population: Number of channel-less threads per title kind.

    Returns:
        The report as markdown.
    """
    judged = [row for row in rows if row["contains_channel"] is not None]
    found = [row for row in judged if row["contains_channel"]]
    total_population = sum(population.values())
    low, high = wilson_interval(len(found), len(judged))
    share = len(found) / len(judged) if judged else 0.0
    input_tokens = sum(row["input_tokens"] for row in rows)
    output_tokens = sum(row["output_tokens"] for row in rows)
    cost = input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE

    lines = [
        f"# Channel judge on a sample of channel-less Slack threads ({MODEL})",
        "",
        f"Sample: {len(rows)} of {total_population:,} threads, stratified by title kind. "
        f"No valid answer: {len(rows) - len(judged)}. Cost: ${cost:.2f} "
        f"({input_tokens:,} input / {output_tokens:,} output tokens).",
        "",
        f"**Reveal their channel: {len(found)}/{len(judged)} = {share:.1%}** "
        f"(95% CI {low:.1%}–{high:.1%}), about {round(share * total_population):,} "
        f"of {total_population:,} threads ({round(low * total_population):,}–"
        f"{round(high * total_population):,}).",
        "",
        "| Title kind | Threads | Sampled | Reveal channel | 95% CI |",
        "|---|---:|---:|---:|---|",
    ]
    for kind, size in population.most_common():
        kind_rows = [row for row in judged if row["title_kind"] == kind]
        kind_found = sum(row["contains_channel"] for row in kind_rows)
        kind_low, kind_high = wilson_interval(kind_found, len(kind_rows))
        kind_share = kind_found / len(kind_rows) if kind_rows else 0.0
        lines.append(
            f"| {kind} | {size:,} | {len(kind_rows)} | {kind_found} ({kind_share:.0%}) "
            f"| {kind_low:.0%}–{kind_high:.0%} |"
        )

    channels = Counter((row["channel"] or "").lstrip("#").lower() for row in found)
    found_channels = ", ".join(f"{name} ({count})" for name, count in channels.most_common())
    lines += ["", f"Channels found: {found_channels or 'none'}"]
    lines += ["", "## Evidence for every channel found", ""]
    lines += [f"- #{row['document_id']} `{row['channel']}`: {row['evidence']!r}" for row in found]

    return "\n".join(lines) + "\n"


def main() -> None:
    """Sample the channel-less threads, judge them and write the report."""
    rng = random.Random(SEED)
    threads = channel_less_threads()
    population = Counter(title_kind(artifact.title) for artifact in threads)
    size = sample_size(len(threads))
    print(f"{len(threads):,} channel-less threads; sampling {size} ({dict(population)})")

    rows = judge_sample(stratified_sample(threads, size, rng))
    RESULTS_FILE.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    text = report(rows, population)
    REPORT_FILE.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
