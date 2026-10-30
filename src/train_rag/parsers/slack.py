"""Parser for Slack threads."""

import ast
import re
from typing import ClassVar

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import OpenText, SlackMessage, SlackThread
from train_rag.parsers.base import DocumentParser
from train_rag.parsers.speakers import (
    BRACED_SPEAKER_LINE,
    BRACKETED_SPEAKER_LINE,
    LOOSE_SPEAKER_LINE,
    speaker_messages,
    strict_speaker_messages,
)
from train_rag.parsers.text import normalize_newlines, unique

KNOWN_CHANNELS = frozenset(
    {
        "all-hands",
        "announcements",
        "architecture",
        "customer-success",
        "design",
        "devex",
        "docs",
        "eng",
        "eng-infra",
        "eng-ml",
        "eng-oncall",
        "eng-platform",
        "eng-releases",
        "eng-runtime",
        "eng-security",
        "eng-sre",
        "finance",
        "general",
        "help",
        "incidents",
        "lunch-plans",
        "marketing",
        "memes",
        "new-hires",
        "partnerships",
        "people-ops",
        "postmortems",
        "product",
        "random",
        "release-war-room",
        "sales",
        "sports",
        "support",
        "vendors",
    }
)
CHANNEL_IN_PATH = re.compile(r"(?:^|/)slack/(?P<channel>[^/]+)/")


class SlackParser(DocumentParser[SlackThread]):
    """Parse a Slack thread: one `speaker: text` or `Name (Role): text` line per message.

    The title is the channel name when it is one of the workspace's channels. Other
    titles are leaked file paths, where the channel is recovered from
    `slack/<channel>/...` when present, or numbers, sentences and one-off slugs. A
    thread with no channel is labelled a group chat rather than flagged. A thread with
    no messages in any format is kept as open text.
    """

    source_type: ClassVar[SourceType] = SourceType.SLACK

    def parse(self, artifact: Artifact) -> SlackThread | OpenText:
        """Parse one Slack thread.

        Args:
            artifact: Raw Slack document.

        Returns:
            The thread with its channel, starter, participants and messages.
        """
        formats: dict[str, str] = {}
        channel = self.try_formats(
            "channel", artifact.title, [self.channel_from_title, self.channel_from_path], formats
        )
        messages = self.try_formats(
            "messages",
            artifact.content,
            [
                self.messages_from_speaker_lines,
                self.messages_from_python_list,
                self.messages_from_loose_speaker_lines,
                self.messages_from_bracketed_speakers,
                self.messages_from_braced_speakers,
            ],
            formats,
        )
        if not messages:
            return self.open_text(artifact)

        return SlackThread(
            doc_id=artifact.id,
            source_type=self.source_type,
            document_id=artifact.document_id,
            formats=formats,
            channel=channel,
            chat_type="channel" if channel else "group_chat",
            thread_starter=messages[0],
            participants=unique([message.speaker for message in messages if message.speaker]),
            messages=messages,
        )

    @staticmethod
    def channel_from_title(title: str) -> str | None:
        """Format 1: the title is the channel name.

        Only `KNOWN_CHANNELS` count: the names used as the title of at least 7 threads.
        Titles that are numbers (`2112233445`), sentences, document ids or one-off topic
        slugs (`compact-cta-toast-ordering`) are not channels.

        Args:
            title: Document title.

        Returns:
            The channel name, or None when the title is not a known channel.
        """
        name = title.strip().lower()

        return name if name in KNOWN_CHANNELS else None

    @staticmethod
    def channel_from_path(title: str) -> str | None:
        """Format 2: the title is a leaked file path like `slack/<channel>/...`.

        Args:
            title: Document title.

        Returns:
            The channel name, or None when the path has no channel.
        """
        match = CHANNEL_IN_PATH.search(title)

        return match.group("channel") if match else None

    @staticmethod
    def messages_from_speaker_lines(content: str) -> list[SlackMessage] | None:
        """Format 1: one `speaker: text` or `Name (Role): text` line per message.

        Lines written the looser way of format 3 (` dave_oncall:`, `maria gonzalez:`)
        also start a message when the speaker's first name is a known one; see
        `known_speaker_line`.

        Args:
            content: Raw thread text.

        Returns:
            The messages in order, or None when no line starts with a speaker.
        """
        return strict_speaker_messages(normalize_newlines(content).splitlines())

    @staticmethod
    def messages_from_python_list(content: str) -> list[SlackMessage] | None:
        """Format 2: the thread is a Python list of `speaker: text` strings.

        Args:
            content: Raw thread text; decoded before newlines are normalized, since
                the list's strings hold escaped line breaks.

        Returns:
            The messages in order, or None when the content is not such a list.
        """
        stripped = content.strip()
        if not stripped.startswith("["):
            return None

        try:
            decoded = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return None

        if not isinstance(decoded, list):
            return None

        lines = [line for item in decoded for line in normalize_newlines(str(item)).splitlines()]

        return strict_speaker_messages(lines)

    @staticmethod
    def messages_from_loose_speaker_lines(content: str) -> list[SlackMessage] | None:
        """Format 3: looser speaker lines than format 1.

        Lines may start with spaces, full names may be lowercase (`maria gonzalez:`),
        and roles may follow without a space (`rebecca(auditor):`) or after a dash
        (`maria - Sales:`).

        Args:
            content: Raw thread text.

        Returns:
            The messages in order, or None when no line starts with a speaker.
        """
        lines = normalize_newlines(content).splitlines()

        return speaker_messages(lines, LOOSE_SPEAKER_LINE.match) or None

    @staticmethod
    def messages_from_bracketed_speakers(content: str) -> list[SlackMessage] | None:
        """Format 4: `[alex] 1789456789: text`, a bracketed speaker and a Unix timestamp.

        The timestamp is optional (`[alex]: text`) and is not kept.

        Args:
            content: Raw thread text.

        Returns:
            The messages in order, or None when no line starts with a bracketed speaker.
        """
        lines = normalize_newlines(content).splitlines()

        return speaker_messages(lines, BRACKETED_SPEAKER_LINE.match) or None

    @staticmethod
    def messages_from_braced_speakers(content: str) -> list[SlackMessage] | None:
        """Format 5: `{Rosie}: text`, a speaker in curly braces.

        Args:
            content: Raw thread text.

        Returns:
            The messages in order, or None when no line starts with a braced speaker.
        """
        lines = normalize_newlines(content).splitlines()

        return speaker_messages(lines, BRACED_SPEAKER_LINE.match) or None
