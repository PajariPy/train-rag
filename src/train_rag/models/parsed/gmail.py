"""Parsed Gmail threads."""

from pydantic import BaseModel, Field

from train_rag.models.parsed.base import ParsedDocument


class EmailMessage(BaseModel):
    """One email with its headers; later replies often omit some headers."""

    sender: str | None = None
    recipients: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    date: str | None = None
    subject: str | None = None
    attachments: list[str] = Field(default_factory=list)
    body: str


class EmailThread(ParsedDocument):
    """An email thread with thread-level metadata collected from its messages.

    Attachments are file names only: the dataset has no attachment files or paths.
    """

    subject: str | None = None
    participants: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
    messages: list[EmailMessage] = Field(default_factory=list)
