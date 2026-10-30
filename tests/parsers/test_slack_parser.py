"""Tests for the Slack parser on real threads."""

from collections.abc import Callable

import pytest

from train_rag.models import Artifact
from train_rag.models.parsed import OpenText, SlackThread
from train_rag.parsers import SlackParser

LoadDocument = Callable[[str], Artifact]


def parse_structured(artifact: Artifact) -> SlackThread:
    """Parse a document that is in one of the formats, not open text.

    Args:
        artifact: Raw document.

    Returns:
        The parsed thread.
    """
    parsed = SlackParser().parse(artifact)
    assert isinstance(parsed, SlackThread)

    return parsed


def test_parses_channel_starter_and_participants(real_document: LoadDocument) -> None:
    """A regular thread yields its channel, first message and speakers in order.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("slack_channel"))

    assert thread.channel == "eng-releases"
    assert thread.thread_starter is not None
    assert thread.thread_starter.speaker == "deploy-bot"
    assert thread.thread_starter.text.startswith("Kicking off staged canary for release v3.19.")
    assert thread.participants[:4] == ["deploy-bot", "olivia", "matt", "devops-bot"]
    assert len(thread.messages) == 15
    assert not thread.flagged


def test_recovers_channel_from_leaked_path_title(real_document: LoadDocument) -> None:
    """A title like `slack/people-ops/<file>.json` still gives the channel.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("slack_path_with_channel"))

    assert thread.channel == "people-ops"
    assert thread.participants[:3] == ["zara", "ivan", "ana"]
    assert not thread.flagged


def test_path_title_without_channel_is_a_group_chat(real_document: LoadDocument) -> None:
    """A bare file-name title means a group chat, while `Name (Role):` speakers still parse.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("slack_path_without_channel"))

    assert thread.channel is None
    assert thread.chat_type == "group_chat"
    assert not thread.flagged
    assert thread.thread_starter is not None
    assert (thread.thread_starter.speaker, thread.thread_starter.role) == ("Olga", "Marketing")
    assert thread.participants[:3] == ["Olga", "Darren", "Felix"]


def test_parses_thread_stored_as_python_list(real_document: LoadDocument) -> None:
    """A thread stored as a list of `speaker: text` strings yields every message.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("slack_python_list"))

    assert len(thread.messages) == 19
    assert thread.participants[:3] == ["kms-rotator-bot", "alex", "maria"]
    assert thread.chat_type == "group_chat"
    assert not thread.flagged


def test_parses_loose_speaker_lines(real_document: LoadDocument) -> None:
    """Lowercase full names and lines starting with a space still start messages.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("slack_loose_speakers"))

    assert len(thread.messages) == 23
    assert thread.participants[:3] == ["maria gonzalez", "tj", "kai"]
    assert not thread.flagged


def test_parses_bracketed_speakers_with_timestamps(real_document: LoadDocument) -> None:
    """`[alex] 1789456789: text` lines start messages; the timestamp is dropped.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("slack_bracketed_speakers"))

    assert len(thread.messages) == 11
    assert thread.participants == ["alex", "meera", "sam", "tyler", "dev-bot"]
    assert thread.messages[0].text.startswith("Heads-up — ran into a parity/shape bug")
    assert not thread.flagged


def test_parses_braced_speakers(real_document: LoadDocument) -> None:
    """`{Rosie}: text` lines start messages.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("slack_braced_speakers"))

    assert len(thread.messages) == 10
    assert thread.participants == ["Rosie", "Sam", "Maria", "Jordan", "Deploybot", "Lee-SRE"]
    assert not thread.flagged


def test_keeps_long_text_without_speakers_as_open_text(real_document: LoadDocument) -> None:
    """300+ characters with no speaker lines are kept whole as useful open text.

    Args:
        real_document: Fixture loader.
    """
    parsed = SlackParser().parse(real_document("slack_unstructured_text"))

    assert isinstance(parsed, OpenText)
    assert parsed.text.startswith("Hey team, customer Acme reported intermittent 429s")
    assert parsed.useful
    assert not parsed.flagged


def test_marks_short_text_without_speakers_not_useful(real_document: LoadDocument) -> None:
    """A thread whose content is just an id is open text that is not useful, and not flagged.

    Args:
        real_document: Fixture loader.
    """
    parsed = SlackParser().parse(real_document("slack_not_useful"))

    assert isinstance(parsed, OpenText)
    assert not parsed.useful
    assert not parsed.flagged


@pytest.mark.parametrize(
    "title",
    [
        "2112233445",
        "PTAL — addressed comments, requested one approve and merge ops slot",
        "compact-cta-toast-ordering",
        "dsid_bf76e7fcd5af49ff9edd187a4d17c7df",
    ],
)
def test_title_that_is_not_a_known_channel_means_group_chat(
    real_document: LoadDocument, title: str
) -> None:
    """Numbers, sentences, one-off slugs and document ids seen as titles are not channels.

    Args:
        real_document: Fixture loader.
        title: A real title that is not a channel.
    """
    document = real_document("slack_channel").model_copy(update={"title": title})

    thread = parse_structured(document)

    assert thread.channel is None
    assert thread.chat_type == "group_chat"
    assert not thread.flagged


@pytest.mark.parametrize(
    ("title", "channel"), [("memes", "memes"), (" Eng-Platform ", "eng-platform")]
)
def test_known_channel_title_is_the_channel(
    real_document: LoadDocument, title: str, channel: str
) -> None:
    """A known channel name, even a small one or in another case, is the channel.

    Args:
        real_document: Fixture loader.
        title: Title as stored.
        channel: Expected channel.
    """
    document = real_document("slack_channel").model_copy(update={"title": title})

    thread = parse_structured(document)

    assert thread.channel == channel
    assert thread.chat_type == "channel"


def test_splits_indented_handles_after_bot_message(real_document: LoadDocument) -> None:
    """` dave_oncall: ...` lines after a bot's message each start a message.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("slack_indented_handles"))

    assert thread.thread_starter is not None
    assert thread.thread_starter.speaker == "incident-bot"
    assert thread.thread_starter.text.endswith("Escalation: on-call SRE.")
    assert [message.speaker for message in thread.messages][:4] == [
        "incident-bot",
        "dave_oncall",
        "maya_eng",
        "ben_platform",
    ]


def test_splits_lowercase_full_names_among_first_names(real_document: LoadDocument) -> None:
    """`alex li:` and `maria gonzalez:` start messages in a thread of `ravi:` lines.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("slack_lowercase_full_names"))

    assert thread.thread_starter is not None
    assert thread.thread_starter.speaker == "alex li"
    assert thread.participants == ["alex li", "maria gonzalez", "ravi", "sam", "ci-bot"]
    assert len(thread.messages) == 15


def test_keeps_lowercase_labels_inside_messages(real_document: LoadDocument) -> None:
    """`form fields:` and `or run:` lines are not speakers: their first word is no name.

    Args:
        real_document: Fixture loader.
    """
    thread = parse_structured(real_document("slack_lowercase_labels"))

    assert thread.participants == ["ian", "maria", "alex", "devex-bot", "lisa"]
    assert "form fields: Name" in thread.messages[6].text
