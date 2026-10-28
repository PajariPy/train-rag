"""Parser for Linear tickets."""

from typing import ClassVar

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import LinearTicket
from train_rag.parsers.base import DocumentParser
from train_rag.parsers.people import comment_participants, parse_comments
from train_rag.parsers.text import leading_text, normalize_newlines, section


class LinearParser(DocumentParser[LinearTicket]):
    """Parse a Linear ticket: its description and dated `comments:`.

    The description comes in several formats, one method each, tried in order: a
    `description:` section, a section with another name, or prose with no key.
    """

    source_type: ClassVar[SourceType] = SourceType.LINEAR

    def parse(self, artifact: Artifact) -> LinearTicket:
        """Parse one Linear ticket.

        Args:
            artifact: Raw Linear document.

        Returns:
            The ticket with description, comments, participants and first activity date.
        """
        text = normalize_newlines(artifact.content)
        formats: dict[str, str] = {}
        description = self.try_formats(
            "description",
            text,
            [
                self.description_section,
                self.body_section,
                self.summary_section,
                self.content_section,
                self.overview_section,
                self.background_section,
                self.incident_summary_section,
                self.problem_statement_section,
                self.objective_section,
                self.unlabeled_description,
            ],
            formats,
        )
        comments = parse_comments(section(text, "comments") or "")
        dates = [comment.posted_at for comment in comments if comment.posted_at is not None]

        return LinearTicket(
            doc_id=artifact.id,
            source_type=self.source_type,
            document_id=artifact.document_id,
            formats=formats,
            description=description,
            comments=comments,
            participants=comment_participants(comments),
            first_activity_at=min(dates, default=None),
            missing_fields=self.missing(description=description),
        )

    @staticmethod
    def description_section(text: str) -> str | None:
        """Format 1: the usual `description:` section.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "description")

    @staticmethod
    def body_section(text: str) -> str | None:
        """Format 2: a `body:` section instead of `description:`.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "body")

    @staticmethod
    def summary_section(text: str) -> str | None:
        """Format 3: a `summary:` section instead of `description:`.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "summary")

    @staticmethod
    def content_section(text: str) -> str | None:
        """Format 4: a `content:` section instead of `description:`.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "content")

    @staticmethod
    def overview_section(text: str) -> str | None:
        """Format 5: an `overview:` section instead of `description:`.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "overview")

    @staticmethod
    def background_section(text: str) -> str | None:
        """Format 6: a `background:` section instead of `description:`.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "background")

    @staticmethod
    def incident_summary_section(text: str) -> str | None:
        """Format 7: an `incident_summary:` section, in incident tickets.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "incident_summary")

    @staticmethod
    def problem_statement_section(text: str) -> str | None:
        """Format 8: a `problem_statement:` section instead of `description:`.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "problem_statement")

    @staticmethod
    def objective_section(text: str) -> str | None:
        """Format 9: an `objective:` section instead of `description:`.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "objective")

    @staticmethod
    def unlabeled_description(text: str) -> str | None:
        """Format 10: prose with no section key, often under labels like `Summary:`.

        Args:
            text: Normalized ticket content.

        Returns:
            The text before the first `key:` section, or None when there is none.
        """
        return leading_text(text)
