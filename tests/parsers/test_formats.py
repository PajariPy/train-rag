"""Tests that each real document is parsed by the format methods made for it."""

from collections.abc import Callable

import pytest

from train_rag.models import Artifact
from train_rag.parsers import default_registry

LoadDocument = Callable[[str], Artifact]

SPEAKERS = {"messages": "messages_from_speaker_lines"}
FIREFLIES_TEXT = {"content": "content_from_text"}
LABELED_ATTENDEES = {"attendees": "attendees_from_labels"}


@pytest.mark.parametrize(
    ("name", "formats"),
    [
        ("slack_channel", {"channel": "channel_from_title", **SPEAKERS}),
        ("slack_path_with_channel", {"channel": "channel_from_path", **SPEAKERS}),
        ("slack_path_without_channel", SPEAKERS),
        ("slack_python_list", {"messages": "messages_from_python_list"}),
        (
            "slack_loose_speakers",
            {"channel": "channel_from_title", "messages": "messages_from_loose_speaker_lines"},
        ),
        (
            "slack_bracketed_speakers",
            {"channel": "channel_from_title", "messages": "messages_from_bracketed_speakers"},
        ),
        (
            "slack_braced_speakers",
            {"channel": "channel_from_title", "messages": "messages_from_braced_speakers"},
        ),
        ("slack_unstructured_text", {"messages": "messages_from_unstructured_text"}),
        ("slack_not_useful", {"channel": "channel_from_title"}),
        ("gmail_json_list", {"messages": "messages_from_json_list"}),
        ("gmail_mixed_quotes_list", {"messages": "messages_from_python_list"}),
        ("gmail_python_list", {"messages": "messages_from_python_list"}),
        ("gmail_numbered_sections", {"messages": "messages_from_numbered_sections"}),
        ("gmail_plain_text", {"messages": "messages_from_text"}),
        ("linear_dated_comments", {"description": "description_section"}),
        ("linear_body_key", {"description": "body_section"}),
        ("linear_background", {"description": "background_section"}),
        ("linear_unlabeled", {"description": "unlabeled_description"}),
        (
            "jira_inline_labels",
            {"description": "description_section", "issue_summary": "inline_issue_summary"},
        ),
        (
            "jira_issue_summary_block",
            {"description": "description_section", "issue_summary": "issue_summary_block"},
        ),
        (
            "jira_summary_long",
            {
                "description": "summary_variant_section",
                "issue_summary": "summary_variant_section",
            },
        ),
        (
            "jira_unlabeled",
            {"description": "unlabeled_description", "issue_summary": "issue_summary_block"},
        ),
        (
            "fireflies_header_line",
            {**FIREFLIES_TEXT, "held_on": "date_from_date_field", **LABELED_ATTENDEES},
        ),
        (
            "fireflies_separate_fields",
            {**FIREFLIES_TEXT, "held_on": "date_from_date_field", **LABELED_ATTENDEES},
        ),
        (
            "fireflies_pipe_header",
            {**FIREFLIES_TEXT, "held_on": "date_from_meeting_header", **LABELED_ATTENDEES},
        ),
        (
            "fireflies_header_label",
            {**FIREFLIES_TEXT, "held_on": "date_from_header_line", **LABELED_ATTENDEES},
        ),
        (
            "fireflies_unlabeled_attendees",
            {
                **FIREFLIES_TEXT,
                "held_on": "date_from_date_field",
                "attendees": "attendees_from_unlabeled_list",
            },
        ),
        (
            "fireflies_python_list",
            {
                "content": "content_from_python_list",
                "held_on": "date_from_header_line",
                **LABELED_ATTENDEES,
            },
        ),
        (
            "fireflies_header_without_colon",
            {**FIREFLIES_TEXT, "held_on": "date_from_header_without_colon", **LABELED_ATTENDEES},
        ),
        (
            "fireflies_recorded_field",
            {**FIREFLIES_TEXT, "held_on": "date_from_recorded_field", **LABELED_ATTENDEES},
        ),
        (
            "fireflies_start_field",
            {**FIREFLIES_TEXT, "held_on": "date_from_start_field", **LABELED_ATTENDEES},
        ),
        (
            "fireflies_without_header",
            {**FIREFLIES_TEXT, "attendees": "attendees_from_transcript"},
        ),
        (
            "github_review_comments",
            {"description": "description_section", "review_comments": "review_comments_section"},
        ),
        (
            "github_review_conversation",
            {
                "description": "description_section",
                "review_comments": "review_conversation_section",
            },
        ),
        (
            "github_pr_body",
            {"description": "pr_body_section", "review_comments": "review_conversation_section"},
        ),
        ("github_without_review", {"description": "description_section"}),
        ("github_unlabeled", {"description": "unlabeled_description"}),
        ("hubspot_record", {}),
    ],
)
def test_records_format_method(
    real_document: LoadDocument, name: str, formats: dict[str, str]
) -> None:
    """The parser records which format method found each multi-format field.

    Args:
        real_document: Fixture loader.
        name: Fixture file stem.
        formats: Expected format method per field.
    """
    parsed = default_registry().parse(real_document(name))

    assert parsed is not None
    assert parsed.formats == formats
