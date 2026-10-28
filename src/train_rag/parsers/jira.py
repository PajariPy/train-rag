"""Parser for Jira tickets."""

import re
from typing import ClassVar

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import JiraTicket
from train_rag.parsers.base import DocumentParser
from train_rag.parsers.people import comment_participants, parse_comments
from train_rag.parsers.text import labeled_value, leading_text, normalize_newlines, section

SUMMARY_VARIANT_KEYS = (
    "summary_long",
    "summary_detail",
    "summary_details",
    "summary_extended",
    "summary_description",
)
ISSUE_SUMMARY_BLOCK = re.compile(
    r"(?im)^[ \t]*issue summary[ \t]*\n(?P<value>.+?)(?:\n[ \t]*\n|\Z)", re.DOTALL
)
DESCRIPTION_LABELS = [
    "Issue summary",
    "Impact",
    "Business impact",
    "Customer impact",
    "Environment",
    "Observed behavior",
    "Observed behaviour",
    "Observed",
    "Expected behavior",
    "Expected",
    "Actual behavior",
    "Actual",
    "Steps to reproduce",
    "Timeline",
    "Time window",
    "Scope",
    "Hypothesis",
    "Initial hypothesis",
    "Request",
    "Symptoms",
    "Background",
    "Attachments",
    "Next steps",
    "Goal",
    "Requirements",
    "Workaround",
    "Root cause",
]


class JiraParser(DocumentParser[JiraTicket]):
    """Parse a Jira ticket.

    The description holds inline labels (`Issue summary: ... Impact: ...
    Environment: ...`), or an `Issue summary` line followed by bullets.
    `Issue summary` serves as the ticket summary for now. The description and the
    issue summary come in several formats, one method each, tried in order.
    """

    source_type: ClassVar[SourceType] = SourceType.JIRA

    def parse(self, artifact: Artifact) -> JiraTicket:
        """Parse one Jira ticket.

        Args:
            artifact: Raw Jira document.

        Returns:
            The ticket with its labeled description fields, comments and participants.
        """
        text = normalize_newlines(artifact.content)
        formats: dict[str, str] = {}
        description = self.try_formats(
            "description",
            text,
            [
                self.description_section,
                self.issue_description_section,
                self.body_section,
                self.content_section,
                self.summary_variant_section,
                self.unlabeled_description,
            ],
            formats,
        )
        labeled_text = description or text
        issue_summary = self.try_formats(
            "issue_summary",
            labeled_text,
            [
                self.inline_issue_summary,
                self.issue_summary_block,
                self.any_case_issue_summary,
                self.inline_summary,
            ],
            formats,
        ) or self.try_formats("issue_summary", text, [self.summary_variant_section], formats)
        comments = parse_comments(section(text, "comments") or "")
        dates = [comment.posted_at for comment in comments if comment.posted_at is not None]

        return JiraTicket(
            doc_id=artifact.id,
            source_type=self.source_type,
            document_id=artifact.document_id,
            formats=formats,
            issue_summary=issue_summary,
            description=description,
            impact=labeled_value(labeled_text, "Impact", DESCRIPTION_LABELS),
            environment=labeled_value(labeled_text, "Environment", DESCRIPTION_LABELS),
            comments=comments,
            participants=comment_participants(comments),
            first_activity_at=min(dates, default=None),
            missing_fields=self.missing(description=description, issue_summary=issue_summary),
        )

    @staticmethod
    def description_section(text: str) -> str | None:
        """Description format 1: the usual `description:` section.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "description")

    @staticmethod
    def issue_description_section(text: str) -> str | None:
        """Description format 2: an `issue_description:` section.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "issue_description")

    @staticmethod
    def body_section(text: str) -> str | None:
        """Description format 3: a `body:` section.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "body")

    @staticmethod
    def content_section(text: str) -> str | None:
        """Description format 4: a `content:` section.

        Args:
            text: Normalized ticket content.

        Returns:
            The section text, or None when the section is absent or empty.
        """
        return section(text, "content")

    @staticmethod
    def summary_variant_section(text: str) -> str | None:
        """A long-summary section such as `summary_long:` or `summary_details:`.

        Used as description format 5, and as the last issue summary format.

        Args:
            text: Normalized ticket content.

        Returns:
            The first such section's text, or None when there is none.
        """
        return next((value for key in SUMMARY_VARIANT_KEYS if (value := section(text, key))), None)

    @staticmethod
    def unlabeled_description(text: str) -> str | None:
        """Description format 6: prose with no section key, e.g. `Requester: ...`.

        Args:
            text: Normalized ticket content.

        Returns:
            The text before the first `key:` section, or None when there is none.
        """
        return leading_text(text)

    @staticmethod
    def inline_issue_summary(text: str) -> str | None:
        """Issue summary format 1: an inline label, `Issue summary: ... Impact: ...`.

        Args:
            text: The description, or the whole ticket when it has no description.

        Returns:
            The issue summary, or None when there is no inline label.
        """
        return labeled_value(text, "Issue summary", DESCRIPTION_LABELS)

    @staticmethod
    def issue_summary_block(text: str) -> str | None:
        """Issue summary format 2: an `Issue summary` line followed by the summary lines.

        Args:
            text: The description, or the whole ticket when it has no description.

        Returns:
            The issue summary, or None when there is no such block.
        """
        block = ISSUE_SUMMARY_BLOCK.search(text)

        return block.group("value").strip() if block else None

    @staticmethod
    def any_case_issue_summary(text: str) -> str | None:
        """Issue summary format 3: the inline label in another case, `Issue Summary:`.

        Args:
            text: The description, or the whole ticket when it has no description.

        Returns:
            The issue summary, or None when there is no such label.
        """
        return labeled_value(text, "Issue summary", DESCRIPTION_LABELS, ignore_case=True)

    @staticmethod
    def inline_summary(text: str) -> str | None:
        """Issue summary format 4: a plain `Summary:` label.

        Args:
            text: The description, or the whole ticket when it has no description.

        Returns:
            The summary, or None when there is no such label.
        """
        return labeled_value(text, "Summary", DESCRIPTION_LABELS)
