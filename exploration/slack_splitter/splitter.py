"""LLM splitter (Gemini): where does each message of a Slack thread start?

The parser splits a thread at lines that start with a speaker. Some threads have several
messages on one line (`... verification. deploy-bot: canary paused ...`), separated by an
escaped line break (backslash-n), or on lines that start with a timestamp
(`(10:02) samir_infra: ...`). The model lists every message's speaker and the first words
of its text, quoted exactly; the thread is then cut at those words, so the message text
stays verbatim.
"""

import re

from google import genai
from google.genai import types
from pydantic import BaseModel

from train_rag.models.parsed import SlackMessage

MODEL = "gemini-3.8-flash"
THINKING_LEVEL = types.ThinkingLevel.LOW
TRAILING_TIMESTAMP = re.compile(r"(?:\\n|\s)*(?:[\[(]\d{1,2}:\d{2}(?::\d{2})?[\])])?(?:\\n|\s)*$")
SYSTEM_PROMPT = """You split Slack threads into messages. The thread was exported as plain \
text and some message boundaries were lost: several messages can sit on one line, be \
separated by a literal "\\n", or start with a timestamp such as "(10:02) samir_infra:".

List every message in order. For each, give the speaker exactly as written (a name, \
handle or bot, without a role in parentheses), the role when one follows the speaker in \
parentheses, and `starts_with`: the first 6 to 10 words of the message text, copied \
character for character from the thread, starting right after "speaker:" (or after \
"speaker (role):").

A new message starts only where someone posts. These are NOT new messages and stay inside \
the message they appear in:
- action items or lists that name owners ("1) ashley: staging spike 2) kevin: cost model");
- someone addressing a person ("Ibrahim: please send the sizes", "@maya: can you check");
- labels and log lines ("Next steps:", "Root cause:", "WARN auth:", "p95 latency: 120ms");
- quoted or pasted text, code and config ("name: api", "max_tokens: 512").

When nothing is merged, list the messages exactly as they already are, one per line."""


class MessageStart(BaseModel):
    """Where one message starts, as the model read it."""

    speaker: str
    role: str | None
    starts_with: str


class Split(BaseModel):
    """The messages of one thread, in order."""

    messages: list[MessageStart]


class SplitResult(BaseModel):
    """A split with the tokens it cost."""

    split: Split | None
    input_tokens: int
    output_tokens: int


def thread_prompt(title: str, content: str) -> str:
    """Build the user prompt for one thread.

    Args:
        title: Thread title.
        content: Raw thread text.

    Returns:
        The prompt text.
    """
    return f"Title: {title}\n\n<thread>\n{content}\n</thread>"


def split_thread(client: genai.Client, title: str, content: str) -> SplitResult:
    """Ask the model where each message of one thread starts.

    Args:
        client: Gemini client.
        title: Thread title.
        content: Raw thread text.

    Returns:
        The split, or None when the model gave no valid answer, plus token usage.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=thread_prompt(title, content),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=Split,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        ),
    )
    usage = response.usage_metadata
    parsed = response.parsed

    return SplitResult(
        split=parsed if isinstance(parsed, Split) else None,
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        output_tokens=(
            (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0) if usage else 0
        ),
    )


def flexible(text: str) -> str:
    """Turn quoted words into a pattern that ignores whitespace and escaped line breaks.

    Args:
        text: Words the model quoted.

    Returns:
        A regular expression matching them in the thread.
    """
    words = [re.escape(word) for word in re.split(r"(?:\s|\\n)+", text.strip()) if word]

    return r"(?:\s|\\n)+".join(words)


def cut_messages(content: str, starts: list[MessageStart]) -> list[SlackMessage] | None:
    """Cut a thread at the quoted message starts.

    Each message runs from its quoted words to the speaker label of the next message.

    Args:
        content: Raw thread text.
        starts: Message starts from the model, in order.

    Returns:
        The messages, or None when a quote is not found in order or a speaker label is
        missing, so a wrong answer never changes the parsed thread.
    """
    positions: list[int] = []
    cursor = 0
    for start in starts:
        match = re.compile(flexible(start.starts_with)).search(content, cursor)
        if match is None or not start.starts_with.strip():
            return None
        positions.append(match.start())
        cursor = match.end()

    messages = []
    for index, (start, position) in enumerate(zip(starts, positions, strict=True)):
        end = len(content)
        if index + 1 < len(starts):
            label = content.lower().rfind(
                starts[index + 1].speaker.lower(), position, positions[index + 1]
            )
            if label == -1:
                return None
            end = label
        text = re.sub(
            r"(?:\\n|\s)*(?:[\[(]\d{1,2}:\d{2}(?::\d{2})?[\])])?(?:\\n|\s)*$",
            "",
            content[position:end],
        )
        messages.append(
            SlackMessage(
                speaker=start.speaker,
                role=start.role,
                text=text.replace("\\n", "\n").strip(),
            )
        )

    return messages
