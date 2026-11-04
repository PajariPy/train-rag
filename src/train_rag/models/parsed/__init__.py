"""Structured fields parsed out of raw documents, one model per source."""

from train_rag.models.parsed.base import Comment, ParsedDocument, Person
from train_rag.models.parsed.fireflies import FirefliesMeeting, InferredDate
from train_rag.models.parsed.github import GithubPullRequest
from train_rag.models.parsed.gmail import EmailMessage, EmailThread
from train_rag.models.parsed.hubspot import HubspotRecord
from train_rag.models.parsed.jira import JiraTicket
from train_rag.models.parsed.linear import LinearTicket
from train_rag.models.parsed.open_text import OpenText
from train_rag.models.parsed.slack import SlackMessage, SlackThread

__all__ = [
    "Comment",
    "EmailMessage",
    "EmailThread",
    "FirefliesMeeting",
    "GithubPullRequest",
    "InferredDate",
    "HubspotRecord",
    "JiraTicket",
    "LinearTicket",
    "OpenText",
    "ParsedDocument",
    "Person",
    "SlackMessage",
    "SlackThread",
]
