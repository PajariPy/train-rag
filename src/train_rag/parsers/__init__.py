"""Deterministic (no LLM) parsers, one replaceable class per source type."""

from collections.abc import Mapping

from train_rag.models.parsed import InferredDate, Person
from train_rag.parsers.base import DocumentParser
from train_rag.parsers.fireflies import FirefliesParser
from train_rag.parsers.github import GithubParser
from train_rag.parsers.gmail import GmailParser
from train_rag.parsers.hubspot import HubspotParser
from train_rag.parsers.jira import JiraParser
from train_rag.parsers.linear import LinearParser
from train_rag.parsers.registry import ParserRegistry
from train_rag.parsers.slack import SlackParser


def default_registry(
    inferred_dates: Mapping[int, InferredDate] | None = None,
    inferred_attendees: Mapping[int, list[Person]] | None = None,
) -> ParserRegistry:
    """Build a registry with the default parser of every parsed source.

    Google Drive and Confluence have no parser yet: they go straight to LLM summaries.

    Args:
        inferred_dates: Meeting dates estimated by the LLM judge, keyed by document id,
            from `load_inferred_dates`; Fireflies meetings without a header date use them.
        inferred_attendees: Speakers listed by an LLM, keyed by document id, from
            `load_inferred_attendees`; Fireflies meetings without attendees use them.

    Returns:
        The registry; call `register` on it to swap a parser.
    """
    return ParserRegistry(
        [
            SlackParser(),
            GmailParser(),
            LinearParser(),
            JiraParser(),
            FirefliesParser(inferred_dates, inferred_attendees),
            GithubParser(),
            HubspotParser(),
        ]
    )


__all__ = [
    "DocumentParser",
    "FirefliesParser",
    "GithubParser",
    "GmailParser",
    "HubspotParser",
    "JiraParser",
    "LinearParser",
    "ParserRegistry",
    "SlackParser",
    "default_registry",
]
