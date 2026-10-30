"""Run the date judge on every Fireflies meeting without a date, as a Gemini batch job.

A batch job costs half the price of normal calls and finishes within 24 hours, usually
much sooner. One request per meeting is written to a JSONL file, uploaded, and run as one
job; the job name is saved so the later steps can find it.

Run:
    uv run python exploration/date_judge/run_batch.py submit    # build, upload and start
    uv run python exploration/date_judge/run_batch.py status    # show the job state
    uv run python exploration/date_judge/run_batch.py collect   # download and report
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from judge import MODEL, SYSTEM_PROMPT, Verdict
from sqlmodel import Session

from train_rag.database import get_engine
from train_rag.models import SourceType
from train_rag.models.parsed import FirefliesMeeting
from train_rag.parsers import FirefliesParser
from train_rag.repository.document import load_artifacts

DATASET = "enterprise_rag_bench"
ENV_FILE = Path(__file__).parents[2] / ".env"
DATA_DIR = Path(__file__).parent / "data"
REQUESTS_FILE = DATA_DIR / "batch_requests.jsonl"
JOB_FILE = DATA_DIR / "batch_job.json"
RESULTS_FILE = DATA_DIR / "batch_results.jsonl"
PREDICTIONS_FILE = DATA_DIR / "batch_predictions.jsonl"
REPORT_FILE = DATA_DIR / "batch_report.md"
INPUT_PRICE = 0.75 / 1_000_000 / 2
OUTPUT_PRICE = 3.75 / 1_000_000 / 2
DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


def undated_meetings() -> list[dict[str, Any]]:
    """Load the meetings whose date the parser could not find.

    Returns:
        One row per meeting with its database id, doc id, title and content.
    """
    parser = FirefliesParser()
    with Session(get_engine()) as session:
        artifacts = load_artifacts(session, DATASET, SourceType.FIREFLIES)

    rows = []
    for artifact in artifacts:
        parsed = parser.parse(artifact)
        if isinstance(parsed, FirefliesMeeting) and parsed.held_on is None:
            rows.append(
                {
                    "document_id": artifact.document_id,
                    "doc_id": artifact.id,
                    "title": artifact.title,
                    "content": artifact.content,
                }
            )

    return rows


def batch_request(meeting: dict[str, Any]) -> dict[str, Any]:
    """Write one meeting as a batch request line, keyed by its database id.

    Args:
        meeting: Row from `undated_meetings`.

    Returns:
        The request in the batch JSONL format, with the same prompt as `judge_meeting`.
    """
    text = f"Title: {meeting['title']}\n\n<meeting>\n{meeting['content']}\n</meeting>"

    return {
        "key": str(meeting["document_id"]),
        "request": {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": Verdict.model_json_schema(),
            },
        },
    }


def submit(client: genai.Client) -> None:
    """Build the requests, upload them and start the batch job.

    Args:
        client: Gemini client.
    """
    if JOB_FILE.exists():
        sys.exit(f"A job was already submitted ({JOB_FILE}); delete that file to start over.")

    meetings = undated_meetings()
    REQUESTS_FILE.write_text(
        "".join(
            json.dumps(batch_request(meeting), ensure_ascii=False) + "\n" for meeting in meetings
        )
    )
    uploaded = client.files.upload(
        file=REQUESTS_FILE,
        config={"display_name": "fireflies-date-judge", "mime_type": "jsonl"},
    )
    job = client.batches.create(
        model=MODEL, src=uploaded.name, config={"display_name": "fireflies-date-judge"}
    )
    JOB_FILE.write_text(
        json.dumps({"job": job.name, "file": uploaded.name, "meetings": len(meetings)}, indent=2)
    )
    print(f"Submitted {job.name} with {len(meetings)} meetings ({job.state}).")


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


def status(client: genai.Client) -> None:
    """Print the state of the saved batch job.

    Args:
        client: Gemini client.
    """
    job = job_state(client)
    print(f"{job.name}: {job.state.name if job.state else 'unknown'}")


def parse_result(line: dict[str, Any]) -> dict[str, Any]:
    """Turn one result line into a prediction row.

    Args:
        line: Decoded line of the results file.

    Returns:
        The document id, verdict fields, token usage and any error.
    """
    response = line.get("response") or {}
    usage = response.get("usageMetadata") or {}
    row: dict[str, Any] = {
        "document_id": int(line["key"]),
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
        "error": json.dumps(line["error"]) if line.get("error") else None,
    }
    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        verdict = Verdict.model_validate_json(text)
    except (KeyError, IndexError, ValueError) as error:
        return {**row, "reveals_date": None, "error": row["error"] or repr(error)}

    return {**row, **verdict.model_dump()}


def report(predictions: list[dict[str, Any]], doc_ids: dict[int, str]) -> str:
    """Summarize the batch results as markdown.

    Args:
        predictions: One row per meeting.
        doc_ids: Dataset doc id per database id, for the examples.

    Returns:
        The report text.
    """
    answered = [row for row in predictions if row["reveals_date"] is not None]
    dated = [row for row in answered if row["reveals_date"] and row["held_on"]]
    input_tokens = sum(row["input_tokens"] for row in predictions)
    output_tokens = sum(row["output_tokens"] for row in predictions)
    cost = input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE
    precisions = Counter(row["precision"] for row in dated)
    years = Counter(row["held_on"][:4] for row in dated)
    median_tokens = statistics.median(row["input_tokens"] for row in predictions)

    lines = [
        f"# Meeting date judge on undated Fireflies meetings ({MODEL}, batch)",
        "",
        "| | Meetings |",
        "|---|---:|",
        f"| Judged | {len(predictions):,} |",
        f"| No valid answer | {len(predictions) - len(answered):,} |",
        f"| Date found | {len(dated):,} ({len(dated) / len(answered):.1%} of answered) |",
        f"| No date in the text | {len(answered) - len(dated):,} |",
        "",
        "| Precision of the date found | Meetings |",
        "|---|---:|",
        *[f"| {name} | {count:,} |" for name, count in precisions.most_common()],
        "",
        "| Year | Meetings |",
        "|---|---:|",
        *[f"| {year} | {count:,} |" for year, count in sorted(years.items())],
        "",
        f"Median tokens per meeting: {median_tokens:,.0f}"
        f" in. Tokens in / out (incl. thinking): {input_tokens:,} / {output_tokens:,}; "
        f"cost at batch prices ≈ ${cost:.2f}.",
        "",
        "## Examples without a date",
        "",
        *[
            f"- `{doc_ids[row['document_id']]}` (#{row['document_id']})"
            for row in answered
            if not row["reveals_date"]
        ][:15],
    ]

    return "\n".join(lines) + "\n"


def collect(client: genai.Client) -> None:
    """Download the results, save one prediction per meeting and write the report.

    Args:
        client: Gemini client.
    """
    job = job_state(client)
    state = job.state.name if job.state else "unknown"
    if state not in DONE_STATES or not job.dest or not job.dest.file_name:
        sys.exit(f"Job is {state}; results are not ready.")

    RESULTS_FILE.write_bytes(client.files.download(file=job.dest.file_name))
    predictions = [
        parse_result(json.loads(line))
        for line in RESULTS_FILE.read_text().splitlines()
        if line.strip()
    ]
    doc_ids = {meeting["document_id"]: meeting["doc_id"] for meeting in undated_meetings()}
    predictions = [{**row, "doc_id": doc_ids.get(row["document_id"])} for row in predictions]
    PREDICTIONS_FILE.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
    )
    text = report(predictions, doc_ids)
    REPORT_FILE.write_text(text)
    print(text)


def main() -> None:
    """Run the step named on the command line."""
    steps = {"submit": submit, "status": status, "collect": collect}
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in steps:
        sys.exit(f"Usage: run_batch.py {{{'|'.join(steps)}}}")

    load_dotenv(ENV_FILE)
    steps[step](genai.Client())


if __name__ == "__main__":
    main()
