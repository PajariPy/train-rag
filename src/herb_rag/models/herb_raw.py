"""Raw HERB JSON schema, used only to validate files before normalizing."""

from datetime import datetime

from pydantic import BaseModel, Field


class RawSlackUser(BaseModel):
    """Sender block of a Slack message."""

    user_id: str = Field(alias="userId")
    timestamp: datetime
    text: str


class RawSlackChannel(BaseModel):
    """Channel a Slack message was posted in."""

    name: str
    channel_id: str = Field(alias="channelID")


class RawSlackBody(BaseModel):
    """Message body of a Slack message."""

    user: RawSlackUser = Field(alias="User")


class RawSlackMessage(BaseModel):
    """A Slack message as stored in a product file."""

    id: str
    channel: RawSlackChannel = Field(alias="Channel")
    message: RawSlackBody = Field(alias="Message")


class RawDocument(BaseModel):
    """An internal document such as a PRD or design doc."""

    id: str
    content: str
    date: datetime
    author: str
    type: str
    document_link: str
    feedback: str | None = None


class RawMeetingTranscript(BaseModel):
    """A meeting transcript with its participants."""

    id: str
    date: datetime
    document_type: str
    participants: list[str]
    transcript: str


class RawMeetingChat(BaseModel):
    """A chat log captured during a meeting."""

    id: str
    text: str


class RawUrl(BaseModel):
    """A shared link with a short description."""

    id: str
    link: str
    description: str


class RawGithubUser(BaseModel):
    """GitHub account reference."""

    login: str


class RawPullRequestReview(BaseModel):
    """A review left on a pull request."""

    state: str
    user: RawGithubUser
    comment: str
    submitted_at: datetime


class RawPullRequest(BaseModel):
    """A GitHub pull request with its reviews."""

    id: str
    title: str
    summary: str
    link: str
    number: str
    state: str
    merged: str
    user: RawGithubUser
    created_at: datetime
    reviews: list[RawPullRequestReview]


class RawAnswerableQuestion(BaseModel):
    """An answerable question with gold answer and cited artifact ids."""

    question: str
    ground_truth: str | list[str]
    citations: list[str]
    type: str


class RawProduct(BaseModel):
    """One product file.

    The `team` and `customers` fields are intentionally not modeled: the
    dataset card reserves them for oracle settings, so RAG must not see them.
    """

    slack: list[RawSlackMessage]
    documents: list[RawDocument]
    meeting_transcripts: list[RawMeetingTranscript]
    meeting_chats: list[RawMeetingChat]
    urls: list[RawUrl]
    prs: list[RawPullRequest]
    answerable_questions: list[RawAnswerableQuestion]
    unanswerable_questions: list[str]
