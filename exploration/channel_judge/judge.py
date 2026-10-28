"""LLM judge (Gemini): does a Slack thread reveal the channel it was posted in?"""

from google import genai
from google.genai import types
from pydantic import BaseModel

MODEL = "gemini-3.8-flash"
SYSTEM_PROMPT = """You review Slack threads exported from a company workspace. The export \
lost the name of the channel each thread was posted in; the title is only a file name.

Decide whether the thread itself reveals the channel this conversation took place \
in. It can show up anywhere: a header or export banner, a participant saying the \
discussion is or stays in a channel, a bot notice about this thread in its channel. The \
name may be written with or without `#`.

Only this conversation's own channel counts. Mentions of other channels are not the \
answer: being asked to go somewhere else ("ask in #support"), copies or pointers \
placed elsewhere ("pinned this in #ops", "cross-posting to #sales", "archived to the \
eng digest"), or plans to move the discussion ("let's move this to #infra"). Also \
ignore user handles, PR or issue numbers, and the file name title.

Return the channel name without `#`, exactly as written, and quote the words that \
show it. When the thread does not reveal its channel, set channel and evidence to null."""


class Verdict(BaseModel):
    """The judge's answer for one thread."""

    contains_channel: bool
    channel: str | None
    evidence: str | None


class JudgeResult(BaseModel):
    """A verdict with the tokens it cost."""

    verdict: Verdict | None
    input_tokens: int
    output_tokens: int
    finish_reason: str | None


def judge_thread(client: genai.Client, title: str, content: str) -> JudgeResult:
    """Ask the model whether one thread reveals its channel.

    Args:
        client: Gemini client.
        title: Thread title (a file name for channel-less threads).
        content: Thread text.

    Returns:
        The parsed verdict, or None when the model gave no valid answer, plus token usage.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Title: {title}\n\n<thread>\n{content}\n</thread>",
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
