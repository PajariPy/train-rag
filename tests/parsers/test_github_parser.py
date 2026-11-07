"""Tests for the GitHub parser on real pull requests."""

from collections.abc import Callable

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import GithubPullRequest, OpenText
from train_rag.parsers import GithubParser

LoadDocument = Callable[[str], Artifact]


def parse_structured(artifact: Artifact) -> GithubPullRequest:
    """Parse a document that is in one of the formats, not open text.

    Args:
        artifact: Raw document.

    Returns:
        The parsed pull request.
    """
    parsed = GithubParser().parse(artifact)
    assert isinstance(parsed, GithubPullRequest)

    return parsed


def test_parses_description_and_review_comments(real_document: LoadDocument) -> None:
    """A `review_comments:` section gives one comment per reviewer line.

    Args:
        real_document: Fixture loader.
    """
    pull_request = parse_structured(real_document("github_review_comments"))

    assert pull_request.description is not None
    assert pull_request.description.startswith("Motivation: tail latency investigations")
    assert pull_request.participants == ["Samir", "Elena", "Tao"]
    assert not pull_request.flagged


def test_splits_inline_review_conversation(real_document: LoadDocument) -> None:
    """Several speakers on one `review_conversation:` line become separate comments.

    Args:
        real_document: Fixture loader.
    """
    pull_request = parse_structured(real_document("github_review_conversation"))

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
    pull_request = parse_structured(real_document("github_without_review"))

    assert pull_request.description is not None
    assert pull_request.review_comments is None
    assert pull_request.participants == []
    assert not pull_request.flagged


def test_reads_pr_body_section(real_document: LoadDocument) -> None:
    """A `pr_body:` section stands in for the description.

    Args:
        real_document: Fixture loader.
    """
    pull_request = parse_structured(real_document("github_pr_body"))

    assert pull_request.description is not None
    assert pull_request.description.startswith("## Summary\nThis PR adds the first cut")
    assert pull_request.review_comments is not None
    assert len(pull_request.review_comments) == 11


def test_reads_unlabeled_description(real_document: LoadDocument) -> None:
    """A pull request written as `Motivation: ...` prose uses it as description.

    Args:
        real_document: Fixture loader.
    """
    pull_request = parse_structured(real_document("github_unlabeled"))

    assert pull_request.description is not None
    assert pull_request.description.startswith("Motivation: CI noise")
    assert not pull_request.flagged


def test_prefers_detailed_description_over_summary(real_document: LoadDocument) -> None:
    """A `summary:` plus `detailed_description:` pull request uses the detailed one.

    Args:
        real_document: Fixture loader.
    """
    pull_request = parse_structured(real_document("github_detailed_description"))

    assert pull_request.formats["description"] == "detailed_description_section"
    assert pull_request.description is not None
    assert pull_request.description.startswith("Context and motivation:")
    assert not pull_request.flagged


def test_keeps_long_document_in_no_format_as_open_text(real_document: LoadDocument) -> None:
    """A long pull request with no description in any format is kept whole as open text.

    Args:
        real_document: Fixture loader.
    """
    parsed = GithubParser().parse(real_document("github_open_text"))

    assert isinstance(parsed, OpenText)
    assert parsed.text.startswith("metric_definitions:")
    assert parsed.useful
    assert not parsed.flagged


def test_marks_short_pull_request_not_useful() -> None:
    """A short pull request in no known format is open text that is not useful, and not flagged."""
    artifact = Artifact(
        id="pr_short", source_type=SourceType.GITHUB, title="t", content="status: open"
    )

    parsed = GithubParser().parse(artifact)

    assert isinstance(parsed, OpenText)
    assert not parsed.useful
    assert not parsed.flagged
