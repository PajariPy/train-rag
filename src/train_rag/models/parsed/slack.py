"""Parsed Slack threads."""

from typing import Literal

from pydantic import BaseModel, Field

from train_rag.models.parsed.base import ParsedDocument


class SlackMessage(BaseModel):
    """One message of a thread; `speaker` is None when the sender is unknown."""

    speaker: str | None
    role: str | None = None
    text: str


class SlackThread(ParsedDocument):
    """A Slack thread with its channel and participants.

    `chat_type` is `group_chat` for a thread with no channel: neither its title nor its
    text names one. These read as short-lived group chats set up to resolve one issue;
    an LLM judge on a random sample of 956 of them found the channel in only 0.4%, so a
    missing channel is a label, not something to review.
    """

    channel: str | None = None
    chat_type: Literal["channel", "group_chat"] = "channel"
    thread_starter: SlackMessage | None = None
    participants: list[str] = Field(default_factory=list)
    messages: list[SlackMessage] = Field(default_factory=list)
