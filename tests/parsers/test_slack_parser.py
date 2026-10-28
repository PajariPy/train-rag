"""Tests for the Slack parser on real threads."""

from collections.abc import Callable

import pytest

from train_rag.models import Artifact
from train_rag.parsers import SlackParser

LoadDocument = Callable[[str], Artifact]


def test_parses_channel_starter_and_participants(real_document: LoadDocument) -> None:
    """A regular thread yields its channel, first message and speakers in order.

    Args:
        real_document: Fixture loader.
    """
    thread = SlackParser().parse(real_document("slack_channel"))

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
    thread = SlackParser().parse(real_document("slack_path_with_channel"))

    assert thread.channel == "people-ops"
    assert thread.participants[:3] == ["zara", "ivan", "ana"]
    assert not thread.flagged


def test_path_title_without_channel_is_a_group_chat(real_document: LoadDocument) -> None:
    """A bare file-name title means a group chat, while `Name (Role):` speakers still parse.

    Args:
        real_document: Fixture loader.
    """
    thread = SlackParser().parse(real_document("slack_path_without_channel"))

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
    thread = SlackParser().parse(real_document("slack_python_list"))

    assert len(thread.messages) == 19
    assert thread.participants[:3] == ["kms-rotator-bot", "alex", "maria"]
    assert thread.chat_type == "group_chat"
    assert not thread.flagged


def test_parses_loose_speaker_lines(real_document: LoadDocument) -> None:
    """Lowercase full names and lines starting with a space still start messages.

    Args:
        real_document: Fixture loader.
    """
    thread = SlackParser().parse(real_document("slack_loose_speakers"))

    assert len(thread.messages) == 23
    assert thread.participants[:3] == ["maria gonzalez", "tj", "kai"]
    assert not thread.flagged


def test_parses_bracketed_speakers_with_timestamps(real_document: LoadDocument) -> None:
    """`[alex] 1789456789: text` lines start messages; the timestamp is dropped.

    Args:
        real_document: Fixture loader.
    """
    thread = SlackParser().parse(real_document("slack_bracketed_speakers"))

    assert len(thread.messages) == 11
    assert thread.participants == ["alex", "meera", "sam", "tyler", "dev-bot"]
    assert thread.messages[0].text.startswith("Heads-up — ran into a parity/shape bug")
    assert not thread.flagged


def test_parses_braced_speakers(real_document: LoadDocument) -> None:
    """`{Rosie}: text` lines start messages.

    Args:
        real_document: Fixture loader.
    """
    thread = SlackParser().parse(real_document("slack_braced_speakers"))

    assert len(thread.messages) == 10
    assert thread.participants == ["Rosie", "Sam", "Maria", "Jordan", "Deploybot", "Lee-SRE"]
    assert not thread.flagged


def test_keeps_long_text_without_speakers_as_one_message(real_document: LoadDocument) -> None:
    """300+ characters with no speaker lines become one message with no known sender.

    The title of this thread is empty, so it is a group chat.

    Args:
        real_document: Fixture loader.
    """
    thread = SlackParser().parse(real_document("slack_unstructured_text"))

    assert len(thread.messages) == 1
    assert thread.messages[0].speaker is None
    assert thread.messages[0].text.startswith("Hey team, customer Acme reported intermittent 429s")
    assert thread.participants == []
    assert thread.useful
    assert thread.chat_type == "group_chat"
    assert not thread.flagged


def test_marks_short_text_without_speakers_not_useful(real_document: LoadDocument) -> None:
    """A thread whose content is just an id is not useful, and not flagged for review.

    Args:
        real_document: Fixture loader.
    """
    thread = SlackParser().parse(real_document("slack_not_useful"))

    assert thread.messages == []
    assert not thread.useful
    assert not thread.flagged


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

    thread = SlackParser().parse(document)

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

    thread = SlackParser().parse(document)

    assert thread.channel == channel
    assert thread.chat_type == "channel"
