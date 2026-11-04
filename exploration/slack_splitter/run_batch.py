"""Run the Slack splitter on every thread with merged messages, as a Gemini batch job.

A thread is sent when one of its parsed messages holds at least two mid-text speaker
labels from a known person or a bot (`... verification. deploy-bot: canary paused`).
Some of those are action items or someone being addressed; the model tells them apart.

Run:
    uv run python exploration/slack_splitter/run_batch.py count     # threads and cost estimate
    uv run python exploration/slack_splitter/run_batch.py submit    # build, upload and start
    uv run python exploration/slack_splitter/run_batch.py status    # show the job state
    uv run python exploration/slack_splitter/run_batch.py collect   # download and split
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from splitter import MODEL, SYSTEM_PROMPT, THINKING_LEVEL, Split, cut_messages, thread_prompt
from sqlmodel import Session

from train_rag.database import get_engine
from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import SlackThread
from train_rag.parsers import SlackParser
from train_rag.parsers.speakers import HANDLE_SEPARATOR, KNOWN_FIRST_NAMES
from train_rag.repository.document import load_artifacts

DATASET = "enterprise_rag_bench"
ENV_FILE = Path(__file__).parents[2] / ".env"
DATA_DIR = Path(__file__).parent / "data"
REQUESTS_FILE = DATA_DIR / "batch_requests.jsonl"
JOB_FILE = DATA_DIR / "batch_job.json"
RESULTS_FILE = DATA_DIR / "batch_results.jsonl"
SPLITS_FILE = DATA_DIR / "batch_splits.jsonl"
INPUT_PRICE = 0.75 / 1_000_000 / 2
OUTPUT_PRICE = 3.75 / 1_000_000 / 2
CHARS_PER_TOKEN = 3.6
DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
INNER_SPEAKER = re.compile(
    r"(?:(?<=[.!?:)\]]\s)|(?<=\\n)|(?<=\n\s))"
    r"(?P<speaker>[A-Za-z][\w.\-]*(?: [A-Za-z][\w.\-]*)?)(?: \([^)\n]{1,40}\))?:\s"
)


def is_known(speaker: str) -> bool:
    """Tell whether a label names a known person or a bot.

    Args:
        speaker: Text before the colon.

    Returns:
        True for a known first name (`dave` in `dave_oncall`) or a bot.
    """
    lowered = speaker.lower()

    return HANDLE_SEPARATOR.split(lowered.split()[0])[0] in KNOWN_FIRST_NAMES or lowered.endswith(
        "bot"
    )


def merged_threads() -> list[Artifact]:
    """Find the threads whose parsed messages hold other speakers' messages.

    Returns:
        The raw threads.
    """
    parser = SlackParser()
    with Session(get_engine()) as session:
        artifacts = load_artifacts(session, DATASET, SourceType.SLACK)

    found = []
    for artifact in artifacts:
        parsed = parser.parse(artifact)
        if not isinstance(parsed, SlackThread):
            continue
        labels = sum(
            1
            for message in parsed.messages
            for match in INNER_SPEAKER.finditer(message.text)
            if is_known(match.group("speaker"))
        )
        if labels >= 2:
            found.append(artifact)

    return found


def batch_request(artifact: Artifact) -> dict[str, Any]:
    """Write one thread as a batch request line, keyed by its database id.

    Args:
        artifact: Raw thread.

    Returns:
        The request in the batch JSONL format, with the same prompt as `split_thread`.
    """
    return {
        "key": str(artifact.document_id),
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": thread_prompt(artifact.title, artifact.content)}],
                }
            ],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": Split.model_json_schema(),
                "thinkingConfig": {"thinkingLevel": THINKING_LEVEL.value},
            },
        },
    }


def count() -> None:
    """Print how many threads would be sent and a rough cost estimate."""
    threads = merged_threads()
    characters = sum(len(artifact.content) for artifact in threads)
    input_tokens = characters / CHARS_PER_TOKEN + 400 * len(threads)
    output_tokens = 600 * len(threads)
    cost = input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE
    print(
        f"{len(threads):,} threads, {characters:,} characters; "
        f"≈ {input_tokens / 1e6:.1f}M tokens in, ≈ ${cost:.2f} at batch prices "
        "(assuming ~600 output and thinking tokens per thread)."
    )


def submit(client: genai.Client) -> None:
    """Build the requests, upload them and start the batch job.

    Args:
        client: Gemini client.
    """
    if JOB_FILE.exists():
        sys.exit(f"A job was already submitted ({JOB_FILE}); delete that file to start over.")

    threads = merged_threads()
    DATA_DIR.mkdir(exist_ok=True)
    REQUESTS_FILE.write_text(
        "".join(json.dumps(batch_request(thread), ensure_ascii=False) + "\n" for thread in threads)
    )
    uploaded = client.files.upload(
        file=REQUESTS_FILE, config={"display_name": "slack-splitter", "mime_type": "jsonl"}
    )
    job = client.batches.create(
        model=MODEL, src=uploaded.name, config={"display_name": "slack-splitter"}
    )
    JOB_FILE.write_text(
        json.dumps({"job": job.name, "file": uploaded.name, "threads": len(threads)}, indent=2)
    )
    print(f"Submitted {job.name} with {len(threads)} threads ({job.state}).")


def job_state(client: genai.Client) -> genai.types.BatchJob:
    """Fetch the saved batch job.

    Args:
        client: Gemini client.

    Returns:
        The job with its current state.
    """
    if not JOB_FILE.exists():
        sys.exit("No job submitted yet; run `submit` first.")

    return client.batches.get(name=json.loads(JOB_FILE.read_text())["job"])


def collect(client: genai.Client) -> None:
    """Download the results, cut each thread and save the splits.

    A thread is saved when the cut works and differs from the parser's split: the model
    both splits merged messages and joins code or log lines (`Host:`, `ERROR:`) that the
    parser took for speakers.

    Args:
        client: Gemini client.
    """
    job = job_state(client)
    state = job.state.name if job.state else "unknown"
    if state not in DONE_STATES or job.dest is None or job.dest.file_name is None:
        sys.exit(f"{job.name} is {state}; try again later.")

    RESULTS_FILE.write_bytes(client.files.download(file=job.dest.file_name))
    threads = {str(artifact.document_id): artifact for artifact in merged_threads()}
    parser = SlackParser()
    rows, input_tokens, output_tokens, failed = [], 0, 0, 0
    for line in RESULTS_FILE.read_text().splitlines():
        result = json.loads(line)
        response = result.get("response") or {}
        usage = response.get("usageMetadata") or {}
        input_tokens += usage.get("promptTokenCount", 0)
        output_tokens += usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        artifact = threads[result["key"]]
        try:
            split = Split.model_validate_json(
                response["candidates"][0]["content"]["parts"][0]["text"]
            )
        except (KeyError, IndexError, ValueError):
            failed += 1
            continue
        messages = cut_messages(artifact.content, split.messages)
        parsed = parser.parse(artifact)
        if messages is None:
            failed += 1
        elif isinstance(parsed, SlackThread) and messages != parsed.messages:
            rows.append(
                {
                    "document_id": artifact.document_id,
                    "messages": [message.model_dump() for message in messages],
                }
            )

    SPLITS_FILE.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    cost = input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE
    print(
        f"{len(threads):,} threads: {len(rows):,} split differently, {failed:,} with no "
        f"usable answer. Tokens in / out: {input_tokens:,} / {output_tokens:,}; ≈ ${cost:.2f}."
    )


def main() -> None:
    """Run the step named on the command line."""
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "count":
        count()
        return

    load_dotenv(ENV_FILE)
    client = genai.Client()
    steps = {"submit": submit, "collect": collect}
    if step == "status":
        job = job_state(client)
        print(f"{job.name}: {job.state.name if job.state else 'unknown'}")
    elif step in steps:
        steps[step](client)
    else:
        sys.exit("Usage: run_batch.py count|submit|status|collect")


if __name__ == "__main__":
    main()
