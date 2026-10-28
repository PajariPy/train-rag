"""Parsed GitHub pull requests."""

from pydantic import Field

from train_rag.models.parsed.base import Comment, ParsedDocument


class GithubPullRequest(ParsedDocument):
    """A pull request with its description and review discussion.

    `review_comments` is None when the pull request has no review discussion, which
    is common and not flagged. The dataset has no creation date for pull requests.
    """

    description: str | None = None
    review_comments: list[Comment] | None = None
    participants: list[str] = Field(default_factory=list)
