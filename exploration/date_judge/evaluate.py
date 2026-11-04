"""Run the date judge on the labelled dataset and report evaluation metrics.

Reads GEMINI_API_KEY from the repository's `.env`.

Run: uv run python exploration/date_judge/evaluate.py
"""

import json
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from judge import MODEL, JudgeResult, judge_meeting

ENV_FILE = Path(__file__).parents[2] / ".env"
DATA_DIR = Path(__file__).parent / "data"
DATASET_FILE = DATA_DIR / "dataset.jsonl"
PREDICTIONS_FILE = DATA_DIR / "predictions.jsonl"
REPORT_FILE = DATA_DIR / "report.md"
WORKERS = 8
INPUT_PRICE = 0.75 / 1_000_000
OUTPUT_PRICE = 3.75 / 1_000_000
ERROR_BANDS = (0, 3, 7, 30)


def days_off(predicted: str | None, true_date: str) -> int | None:
    """Signed distance in days from the true date to the predicted one.

    Args:
        predicted: Predicted date as YYYY-MM-DD, or None.
        true_date: True meeting date as YYYY-MM-DD.

    Returns:
        Predicted minus true in days, or None when there is no valid prediction.
    """
    if not predicted:
        return None
    try:
        return (date.fromisoformat(predicted) - date.fromisoformat(true_date)).days
    except ValueError:
        return None


def predict(client: genai.Client, case: dict[str, Any]) -> dict[str, Any]:
    """Judge one case and attach the prediction to it.

    Args:
        client: Gemini client.
        case: Dataset row.

    Returns:
        The row with the verdict, its error in days, token usage and finish reason.
    """
    result: JudgeResult = judge_meeting(client, case["title"], case["content"])
    verdict = result.verdict
    predicted_date = verdict.held_on if verdict and verdict.reveals_date else None

    return {
        **{key: value for key, value in case.items() if key != "content"},
        "predicted": verdict.reveals_date if verdict else None,
        "predicted_date": predicted_date,
        "precision": verdict.precision if verdict else None,
        "evidence": verdict.evidence if verdict else None,
        "days_off": days_off(predicted_date, case["true_date"]),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "finish_reason": result.finish_reason,
    }


def error_table(rows: list[dict[str, Any]]) -> list[str]:
    """Summarize how far the predicted dates are from the true ones.

    Args:
        rows: Cases with a date that the judge gave a date for.

    Returns:
        Markdown lines with the share within each error band and the median error.
    """
    errors = [abs(row["days_off"]) for row in rows if row["days_off"] is not None]
    if not errors:
        return ["No dates predicted."]

    signed = [row["days_off"] for row in rows if row["days_off"] is not None]
    lines = ["| Within | Cases |", "|---|---:|"]
    lines += [
        f"| {'exact day' if band == 0 else f'{band} days'} | "
        f"{sum(error <= band for error in errors)}/{len(errors)} |"
        for band in ERROR_BANDS
    ]
    lines += [
        "",
        f"Median absolute error: {statistics.median(errors):g} days; median signed error "
        f"(predicted − true): {statistics.median(signed):+g} days; worst: {max(errors)} days.",
    ]

    return lines


def report(predictions: list[dict[str, Any]]) -> str:
    """Compute detection and date-error metrics as a markdown report.

    Args:
        predictions: Rows with labels and predictions.

    Returns:
        The report text.
    """
    synthetic = [row for row in predictions if row["group"] == "synthetic"]
    real = [row for row in predictions if row["group"] == "real"]
    outcomes = Counter(
        (row["has_date"], bool(row["predicted"]))
        for row in synthetic
        if row["predicted"] is not None
    )
    true_pos, false_neg = outcomes[True, True], outcomes[True, False]
    false_pos, true_neg = outcomes[False, True], outcomes[False, False]
    judged = sum(outcomes.values())
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    unanswered = sum(row["predicted"] is None for row in predictions)
    input_tokens = sum(row["input_tokens"] for row in predictions)
    output_tokens = sum(row["output_tokens"] for row in predictions)
    cost = input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE
    found = [row for row in synthetic if row["has_date"] and row["predicted"]]

    lines = [
        f"# Meeting date judge evaluation ({MODEL})",
        "",
        f"{len(synthetic)} synthetic cases: {sum(row['has_date'] for row in synthetic)} with a "
        f"hidden date clue, {sum(not row['has_date'] for row in synthetic)} without. "
        f"{len(real)} real cases with their own dates left in the text. "
        f"No valid answer: {unanswered}.",
        "",
        "## Synthetic: does the text reveal the date?",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {(true_pos + true_neg) / judged:.1%} |",
        f"| Precision | {precision:.1%} |",
        f"| Recall | {recall:.1%} |",
        f"| F1 | {f1:.1%} |",
        "",
        "| | Judged: has date | Judged: no date |",
        "|---|---:|---:|",
        f"| **Has date** | {true_pos} | {false_neg} |",
        f"| **No date** | {false_pos} | {true_neg} |",
        "",
        "| Where the clue was hidden | Found | Exact day |",
        "|---|---:|---:|",
    ]
    placements = Counter(row["placement"] for row in synthetic if row["has_date"])
    found_by_placement = Counter(row["placement"] for row in found)
    exact_by_placement = Counter(row["placement"] for row in found if row["days_off"] == 0)
    lines += [
        f"| {name} | {found_by_placement[name]}/{total} | {exact_by_placement[name]}/{total} |"
        for name, total in placements.items()
    ]
    negatives = Counter(row["placement"] for row in synthetic if not row["has_date"])
    wrongly_found = Counter(
        row["placement"] for row in synthetic if not row["has_date"] and row["predicted"]
    )
    lines += ["", "| Meeting without a date clue | Wrongly given a date |", "|---|---:|"]
    lines += [f"| {name} | {wrongly_found[name]}/{total} |" for name, total in negatives.items()]
    lines += ["", "### Date error on found clues", "", *error_table(found)]

    real_found = [row for row in real if row["predicted"]]
    lines += [
        "",
        "## Real: dates as they occur in the data",
        "",
        f"The judge gave a date for {len(real_found)}/{len(real)} meetings.",
        "",
        *error_table(real_found),
        "",
        "| Case | True date | Predicted | Precision | Days off | Evidence |",
        "|---|---|---|---|---:|---|",
    ]
    lines += [
        f"| `{row['id']}` | {row['true_date']} | {row['predicted_date']} | {row['precision']} "
        f"| {row['days_off'] if row['days_off'] is not None else '—'} "
        f"| {(row['evidence'] or '').replace('|', '/')[:90]} |"
        for row in real
    ]

    mistakes = [
        row
        for row in synthetic
        if bool(row["predicted"]) != row["has_date"]
        or (
            row["has_date"]
            and row["predicted"]
            and (row["days_off"] is None or abs(row["days_off"]) > 7)
        )
    ]
    if mistakes:
        lines += ["", "## Synthetic mistakes", ""]
        lines += [
            f"- `{row['id']}` ({row['placement']}): expected `{row['held_on']}`, got "
            f"`{row['predicted_date']}` — {row['evidence']!r}"
            for row in mistakes
        ]
    lines += [
        "",
        f"Tokens in / out (incl. thinking): {input_tokens:,} / {output_tokens:,}; "
        f"cost ≈ ${cost:.3f}.",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    """Judge every case, save predictions and print the report."""
    cases = [json.loads(line) for line in DATASET_FILE.read_text().splitlines()]
    load_dotenv(ENV_FILE)
    client = genai.Client()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        predictions = list(pool.map(lambda case: predict(client, case), cases))

    PREDICTIONS_FILE.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
    )
    text = report(predictions)
    REPORT_FILE.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
