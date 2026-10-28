"""Tests for the GitHub parser on real pull requests."""

from collections.abc import Callable

from train_rag.models import Artifact
from train_rag.parsers import GithubParser

LoadDocument = Callable[[str], Artifact]


def test_parses_description_and_review_comments(real_document: LoadDocument) -> None:
    """A `review_comments:` section gives one comment per reviewer line.

    Args:
        real_document: Fixture loader.
    """
    pull_request = GithubParser().parse(real_document("github_review_comments"))

    assert pull_request.description is not None
    assert pull_request.description.startswith("Motivation: tail latency investigations")
    assert pull_request.participants == ["Samir", "Elena", "Tao"]
    assert not pull_request.flagged


def test_splits_inline_review_conversation(real_document: LoadDocument) -> None:
    """Several speakers on one `review_conversation:` line become separate comments.

    Args:
        real_document: Fixture loader.
    """
    pull_request = GithubParser().parse(real_document("github_review_conversation"))

    assert pull_request.review_comments is not None
    authors = [comment.author for comment in pull_request.review_comments]

    assert authors[:4] == ["Luca", "Maya", "Priya", "Maya"]
    assert pull_request.participants == ["Luca", "Maya", "Priya", "Owen"]
    assert pull_request.review_comments[0].text.startswith("This looks promising.")
    assert not pull_request.flagged


def test_pull_request_without_review_is_not_flagged(real_document: LoadDocument) -> None:
    """A pull request with no review discussion keeps None and is not flagged.

    Args:
        real_document: Fixture loader.
    """
    pull_request = GithubParser().parse(real_document("github_without_review"))

    assert pull_request.description is not None
    assert pull_request.review_comments is None
    assert pull_request.participants == []
    assert not pull_request.flagged


def test_reads_pr_body_section(real_document: LoadDocument) -> None:
    """A `pr_body:` section stands in for the description.

    Args:
        real_document: Fixture loader.
    """
    pull_request = GithubParser().parse(real_document("github_pr_body"))

    assert pull_request.description is not None
    assert pull_request.description.startswith("## Summary\nThis PR adds the first cut")
    assert pull_request.review_comments is not None
    assert len(pull_request.review_comments) == 11


def test_reads_unlabeled_description(real_document: LoadDocument) -> None:
    """A pull request written as `Motivation: ...` prose uses it as description.

    Args:
        real_document: Fixture loader.
    """
    pull_request = GithubParser().parse(real_document("github_unlabeled"))

    assert pull_request.description is not None
    assert pull_request.description.startswith("Motivation: CI noise")
    assert not pull_request.flagged
