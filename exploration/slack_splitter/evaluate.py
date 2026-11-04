"""Evaluate the Slack splitter on the test set, next to the parser as a baseline.

A message counts as found when its speaker and the start of its text match a true
message. Per case: exact when every message is found and nothing extra; per message:
precision and recall of the message starts. 20 real merged threads are split too, for
reading: the report shows how many messages the parser and the model find.

Reads GEMINI_API_KEY from the repository's `.env`.

Run: uv run python exploration/slack_splitter/evaluate.py
"""

import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from run_batch import INPUT_PRICE, OUTPUT_PRICE, merged_threads
from splitter import MODEL, SplitResult, cut_messages, split_thread

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import SlackMessage, SlackThread
from train_rag.parsers import SlackParser

ENV_FILE = Path(__file__).parents[2] / ".env"
DATA_DIR = Path(__file__).parent / "data"
DATASET_FILE = DATA_DIR / "dataset.jsonl"
PREDICTIONS_FILE = DATA_DIR / "predictions.jsonl"
REPORT_FILE = DATA_DIR / "report.md"
REAL_CASES = 20
WORKERS = 8
SEED = 3


def key(speaker: str, text: str) -> tuple[str, str]:
    """Identify a message by its speaker and the first 40 characters of its text.

    Args:
        speaker: Message speaker.
        text: Message text.

    Returns:
        The lowercased speaker and the whitespace-normalized text start.
    """
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " ")).strip()

    return speaker.lower(), normalized[:40]


def score(predicted: list[SlackMessage] | None, truth: list[list[str]]) -> dict[str, Any]:
    """Compare predicted messages with the true ones.

    Args:
        predicted: Messages found, or None when there is no usable answer.
        truth: True (speaker, text) messages.

    Returns:
        Counts of found, predicted and true messages, and whether the case is exact.
    """
    true_keys = [key(speaker, text) for speaker, text in truth]
    predicted_keys = [key(message.speaker or "", message.text) for message in predicted or []]
    found = len(set(predicted_keys) & set(true_keys))

    return {
        "found": found,
        "predicted": len(predicted_keys),
        "true": len(true_keys),
        "exact": predicted_keys == true_keys,
    }


def run_case(client: genai.Client, case: dict[str, Any]) -> dict[str, Any]:
    """Split one test case with the parser and the model.

    Args:
        client: Gemini client.
        case: Test case from the dataset.

    Returns:
        The case id and kind, both scores and the model's token usage.
    """
    artifact = Artifact(
        id=case["case_id"],
        source_type=SourceType.SLACK,
        title=case["title"],
        content=case["content"],
    )
    parsed = SlackParser().parse(artifact)
    result = split_thread(client, case["title"], case["content"])
    messages = cut_messages(case["content"], result.split.messages) if result.split else None

    return {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "parser": score(
            parsed.messages if isinstance(parsed, SlackThread) else None, case["messages"]
        ),
        "model": score(messages, case["messages"]),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


def table(rows: list[dict[str, Any]], who: str) -> list[str]:
    """Summarize one splitter's scores per kind of case.

    Args:
        rows: Results of `run_case`.
        who: `parser` or `model`.

    Returns:
        Markdown table lines.
    """
    kinds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        kinds[row["kind"]].append(row[who])
    lines = ["| Kind | Exact | Precision | Recall |", "|---|---:|---:|---:|"]
    for kind, scores in [*kinds.items(), ("all", [row[who] for row in rows])]:
        found = sum(item["found"] for item in scores)
        predicted = sum(item["predicted"] for item in scores)
        true = sum(item["true"] for item in scores)
        exact = sum(item["exact"] for item in scores)
        lines.append(
            f"| {kind} | {exact}/{len(scores)} | {found / predicted if predicted else 0:.1%} "
            f"| {found / true:.1%} |"
        )

    return lines


def real_rows(client: genai.Client) -> list[dict[str, Any]]:
    """Split a random sample of real merged threads.

    Args:
        client: Gemini client.

    Returns:
        Per thread: the message counts of the parser and the model, and usage.
    """
    sample = random.Random(SEED).sample(merged_threads(), REAL_CASES)

    def run(artifact: Artifact) -> dict[str, Any]:
        parsed = SlackParser().parse(artifact)
        result: SplitResult = split_thread(client, artifact.title, artifact.content)
        messages = cut_messages(artifact.content, result.split.messages) if result.split else None

        return {
            "document_id": artifact.document_id,
            "parser": len(parsed.messages) if isinstance(parsed, SlackThread) else 0,
            "model": len(messages) if messages is not None else None,
            "speakers": [message.speaker for message in messages or []],
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(run, sample))


def main() -> None:
    """Run the parser and the model on every case and write the report."""
    load_dotenv(ENV_FILE)
    client = genai.Client()
    cases = [json.loads(line) for line in DATASET_FILE.read_text().splitlines()]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(lambda case: run_case(client, case), cases))
    real = real_rows(client)
    PREDICTIONS_FILE.write_text("".join(json.dumps(row) + "\n" for row in [*rows, *real]))

    input_tokens = sum(row["input_tokens"] for row in [*rows, *real])
    output_tokens = sum(row["output_tokens"] for row in [*rows, *real])
    cost = input_tokens * INPUT_PRICE * 2 + output_tokens * OUTPUT_PRICE * 2
    lines = [
        f"# Slack splitter evaluation ({MODEL})",
        "",
        f"{len(rows)} test cases built from clean threads; {len(real)} real merged threads.",
        "",
        "## Parser (baseline)",
        "",
        *table(rows, "parser"),
        "",
        "## Model",
        "",
        *table(rows, "model"),
        "",
        "## Real merged threads",
        "",
        "| Thread | Parser messages | Model messages | Speakers |",
        "|---|---:|---:|---|",
        *[
            f"| #{row['document_id']} | {row['parser']} | {row['model'] or 'no cut'} "
            f"| {', '.join(dict.fromkeys(row['speakers']))} |"
            for row in real
        ],
        "",
        f"Tokens in / out (incl. thinking): {input_tokens:,} / {output_tokens:,}; "
        f"≈ {output_tokens / len([*rows, *real]):,.0f} out per thread; cost ≈ ${cost:.2f}.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n")
    print(REPORT_FILE.read_text())


if __name__ == "__main__":
    main()
