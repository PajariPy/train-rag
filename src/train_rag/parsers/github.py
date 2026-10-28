"""Parser for GitHub pull requests."""

from typing import ClassVar

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import GithubPullRequest
from train_rag.parsers.base import DocumentParser
from train_rag.parsers.people import comment_participants, parse_comments
from train_rag.parsers.text import leading_text, normalize_newlines, section


class GithubParser(DocumentParser[GithubPullRequest]):
    """Parse a pull request: its description and review discussion.

    The description and the review discussion appear under several section names,
    one method each; the first one present is used. Only a missing description is
    flagged: many pull requests have no review discussion, and that is left as None.
    """

    source_type: ClassVar[SourceType] = SourceType.GITHUB

    def parse(self, artifact: Artifact) -> GithubPullRequest:
        """Parse one pull request.

        Args:
            artifact: Raw GitHub document.

        Returns:
            The pull request with its description, review comments and participants.
        """
        text = normalize_newlines(artifact.content)
        formats: dict[str, str] = {}
        description = self.try_formats(
            "description",
            text,
            [
                self.description_section,
                self.body_section,
                self.pr_body_section,
                self.content_section,
                self.unlabeled_description,
            ],
            formats,
        )
        review = self.try_formats(
            "review_comments",
            text,
            [
                self.review_comments_section,
                self.review_conversation_section,
                self.review_thread_section,
                self.review_threads_section,
                self.review_discussion_section,
                self.reviews_section,
                self.conversation_section,
                self.discussion_section,
                self.comments_section,
            ],
            formats,
        )
        review_comments = parse_comments(review, inline_speakers=True) if review else None

        return GithubPullRequest(
            doc_id=artifact.id,
            source_type=self.source_type,
            document_id=artifact.document_id,
            formats=formats,
            description=description,
            review_comments=review_comments,
            participants=comment_participants(review_comments or []),
            missing_fields=self.missing(description=description),
        )

    @staticmethod
    def description_section(text: str) -> str | None:
        """Format 1: the usual `description:` section.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "description")

    @staticmethod
    def body_section(text: str) -> str | None:
        """Format 2: a `body:` section instead of `description:`.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "body")

    @staticmethod
    def pr_body_section(text: str) -> str | None:
        """Format 3: a `pr_body:` section instead of `description:`.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "pr_body")

    @staticmethod
    def content_section(text: str) -> str | None:
        """Format 4: a `content:` section instead of `description:`.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "content")

    @staticmethod
    def unlabeled_description(text: str) -> str | None:
        """Format 5: prose with no section key, e.g. `Motivation: ...`.

        Args:
            text: Normalized pull request content.

        Returns:
            The text before the first `key:` section, or None when there is none.
        """
        return leading_text(text)

    @staticmethod
    def review_comments_section(text: str) -> str | None:
        """Format 1: the usual `review_comments:` section.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "review_comments")

    @staticmethod
    def review_conversation_section(text: str) -> str | None:
        """Format 2: a `review_conversation:` section.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "review_conversation")

    @staticmethod
    def review_thread_section(text: str) -> str | None:
        """Format 3: a `review_thread:` section.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "review_thread")

    @staticmethod
    def review_threads_section(text: str) -> str | None:
        """Format 4: a `review_threads:` section.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "review_threads")

    @staticmethod
    def review_discussion_section(text: str) -> str | None:
        """Format 5: a `review_discussion:` section.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "review_discussion")

    @staticmethod
    def reviews_section(text: str) -> str | None:
        """Format 6: a `reviews:` section.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "reviews")

    @staticmethod
    def conversation_section(text: str) -> str | None:
        """Format 7: a `conversation:` section.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "conversation")

    @staticmethod
    def discussion_section(text: str) -> str | None:
        """Format 8: a `discussion:` section.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "discussion")

    @staticmethod
    def comments_section(text: str) -> str | None:
        """Format 9: a `comments:` section.

        Args:
            text: Normalized pull request content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "comments")
