"""LLM judge (Gemini): when was a Fireflies meeting held, judging by its text alone?"""

from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

MODEL = "gemini-3.8-flash"
SYSTEM_PROMPT = """You review meeting transcripts exported from Fireflies. The export lost \
the header with the date the meeting was held.

Decide whether the text itself lets you work out, at least approximately, the date the \
meeting took place, and estimate it. Useful clues tie a date to the meeting day:
- someone states today's date or the current week or month;
- a relative reference anchored to a dated event ("the release went out on 2026-03-02, two \
days ago", "the 2026-05-01 deadline is a week from today");
- file names or links stamped with today's or yesterday's date;
- action items due on given dates: they are set at the meeting, usually due a few days to \
two weeks later, so the meeting took place shortly before the earliest due date.

Dates that say nothing about when the meeting happened are not clues: events years in the \
past ("back in 2023"), expiry or renewal dates years ahead, version numbers, or dates \
with no link to the present. When the only dates are like that, or there are none, the \
text does not reveal the date.

Return the estimated date as YYYY-MM-DD, how precise it is ("day" when the text pins the \
exact day, "week" when it is within about a week, "month" when only the month is clear), \
and quote the words that show it. When the text does not reveal the date, set held_on, \
precision and evidence to null."""


class Verdict(BaseModel):
    """The judge's answer for one meeting."""

    reveals_date: bool
    held_on: str | None
    precision: Literal["day", "week", "month"] | None
    evidence: str | None


class JudgeResult(BaseModel):
    """A verdict with the tokens it cost."""

    verdict: Verdict | None
    input_tokens: int
    output_tokens: int
    finish_reason: str | None


def judge_meeting(client: genai.Client, title: str, content: str) -> JudgeResult:
    """Ask the model when one meeting was held.

    Args:
        client: Gemini client.
        title: Meeting title.
        content: Meeting text.

    Returns:
        The parsed verdict, or None when the model gave no valid answer, plus token usage.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Title: {title}\n\n<meeting>\n{content}\n</meeting>",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=Verdict,
        ),
    )
    usage = response.usage_metadata
    parsed = response.parsed
    candidate = response.candidates[0] if response.candidates else None

    return JudgeResult(
        verdict=parsed if isinstance(parsed, Verdict) else None,
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        output_tokens=(
            (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0) if usage else 0
        ),
        finish_reason=str(candidate.finish_reason) if candidate else None,
    )
