"""Parser for Slack threads."""

import ast
import re
from typing import ClassVar

from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import SlackMessage, SlackThread
from train_rag.parsers.base import DocumentParser
from train_rag.parsers.text import normalize_newlines, unique

SPEAKER_LINE = re.compile(
    r"^(?P<speaker>[A-Za-z][\w.\-]*(?: [A-Z][\w.\-]*){0,2})"
    r"(?: \((?P<role>[^)\n]{1,40})\))?:\s?(?P<text>.*)$"
)
LOOSE_SPEAKER_LINE = re.compile(
    r"^\s*(?P<speaker>[A-Za-z][\w.\-]*(?: [A-Za-z][\w.\-]*){0,2})\s*"
    r"(?:\((?P<role>[^)\n]{1,40})\)|\s-\s(?P<dash_role>[A-Za-z][\w &/-]{0,30}))?"
    r":\s?(?P<text>.*)$"
)
BRACKETED_SPEAKER_LINE = re.compile(
    r"^\s*\[(?P<speaker>[A-Za-z][\w.\- ]{0,40})\]\s*(?:\d{9,10}(?:\.\d+)?)?\s*:\s?(?P<text>.*)$"
)
BRACED_SPEAKER_LINE = re.compile(r"^\s*\{(?P<speaker>[A-Za-z][\w.\- ]{0,40})\}\s*:\s?(?P<text>.*)$")
MIN_USEFUL_CHARS = 300
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
    no messages in any format is marked not useful.
    """

    source_type: ClassVar[SourceType] = SourceType.SLACK

    def parse(self, artifact: Artifact) -> SlackThread:
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
        messages = (
            self.try_formats(
                "messages",
                artifact.content,
                [
                    self.messages_from_speaker_lines,
                    self.messages_from_python_list,
                    self.messages_from_loose_speaker_lines,
                    self.messages_from_bracketed_speakers,
                    self.messages_from_braced_speakers,
                    self.messages_from_unstructured_text,
                ],
                formats,
            )
            or []
        )
        thread_starter = messages[0] if messages else None
        useful = bool(messages)

        return SlackThread(
            doc_id=artifact.id,
            source_type=self.source_type,
            document_id=artifact.document_id,
            formats=formats,
            channel=channel,
            chat_type="channel" if channel else "group_chat",
            thread_starter=thread_starter,
            participants=unique([message.speaker for message in messages if message.speaker]),
            messages=messages,
            missing_fields=self.missing(messages=messages) if useful else [],
            useful=useful,
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

        Args:
            content: Raw thread text.

        Returns:
            The messages in order, or None when no line starts with a speaker.
        """
        return speaker_messages(normalize_newlines(content).splitlines(), SPEAKER_LINE) or None

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

        return speaker_messages(lines, SPEAKER_LINE) or None

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

        return speaker_messages(lines, LOOSE_SPEAKER_LINE) or None

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

        return speaker_messages(lines, BRACKETED_SPEAKER_LINE) or None

    @staticmethod
    def messages_from_braced_speakers(content: str) -> list[SlackMessage] | None:
        """Format 5: `{Rosie}: text`, a speaker in curly braces.

        Args:
            content: Raw thread text.

        Returns:
            The messages in order, or None when no line starts with a braced speaker.
        """
        lines = normalize_newlines(content).splitlines()

        return speaker_messages(lines, BRACED_SPEAKER_LINE) or None

    @staticmethod
    def messages_from_unstructured_text(content: str) -> list[SlackMessage] | None:
        """Format 6: text with no speaker lines, kept as one message with no known sender.

        Shorter text (an id, a list of names, `(end)`) is not worth indexing: no
        message is returned, and the thread is marked not useful.

        Args:
            content: Raw thread text.

        Returns:
            One message with no speaker, or None when the text is under
            `MIN_USEFUL_CHARS` characters.
        """
        text = normalize_newlines(content).strip()
        if len(text) < MIN_USEFUL_CHARS:
            return None

        return [SlackMessage(speaker=None, text=text)]


def speaker_messages(lines: list[str], speaker_line: re.Pattern[str]) -> list[SlackMessage]:
    """Split lines into messages; lines without a speaker continue the previous one.

    Args:
        lines: Thread lines.
        speaker_line: Pattern of a line that starts a message, with `speaker`, `text`
            and optional `role` or `dash_role` groups.

    Returns:
        The messages in order.
    """
    messages: list[SlackMessage] = []
    for line in lines:
        match = speaker_line.match(line)
        if match is not None:
            groups = match.groupdict()
            messages.append(
                SlackMessage(
                    speaker=match.group("speaker"),
                    role=groups.get("role") or groups.get("dash_role"),
                    text=match.group("text").strip(),
                )
            )
        elif messages and line.strip():
            messages[-1].text = f"{messages[-1].text}\n{line}"

    return messages
