"""Clean the questions an LLM writes for the training set."""

import re

MAX_CHARS = 400
PREFIX = re.compile(r"^(query|question)\s*:\s*", re.IGNORECASE)


def clean_question(raw: str) -> str | None:
    """Turn a model's answer into one clean question.

    Takes the first non-empty line, then strips markdown emphasis, surrounding quotes and a
    "Query:" or "Question:" prefix.

    Args:
        raw: The model's answer.

    Returns:
        The question, or None if nothing usable is left or it's longer than `MAX_CHARS`.
    """
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None
    question = lines[0].strip("*_` ").strip().strip("\"'").strip()
    question = PREFIX.sub("", question).strip().strip("\"'").strip()
    if not question or len(question) > MAX_CHARS:
        return None

    return question
