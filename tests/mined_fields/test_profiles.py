"""Tests for the mined fields model, the profile texts and the result cards."""

from train_rag.mined_fields.profiles import field_card, plain_card, profiles, source_line
from train_rag.models.mined_fields import MinedFields


def mined() -> MinedFields:
    """Build the fields of a small incident thread.

    Returns:
        The fields.
    """
    return MinedFields.from_label(
        {
            "fields": {
                "topic": ["429s after burst limit change"],
                "concept": ["rate limiting"],
                "bug": ["429 errors"],
                "fix": ["idempotency keys", "raise burst", "revert"],
                "client": ["acme-inc"],
                "person": ["ana"],
                "made_up_field": ["dropped"],
            },
            "labels": {
                "thread_kind": "incident",
                "status": "made up",
                "area": ["API and SDK", "not an area", "API and SDK", "billing and pricing"],
            },
        }
    )


def test_unknown_values_are_dropped_and_missing_fields_filled() -> None:
    """Fields outside the schema, unknown labels and repeated areas go; missing fields are []."""
    fields = mined()
    assert "made_up_field" not in fields.fields
    assert fields.fields["cause"] == []
    assert fields.thread_kind == "incident"
    assert fields.status is None
    assert fields.areas == ["API and SDK", "billing and pricing"]


def test_profiles_join_their_fields() -> None:
    """Each profile joins its fields: subject, problem and names."""
    texts = profiles(mined())
    assert texts.subject == "429s after burst limit change; rate limiting"
    assert texts.problem == "429 errors; idempotency keys; raise burst; revert"
    assert texts.names == "ana; acme-inc"


def test_empty_fields_give_empty_profiles() -> None:
    """A document with nothing mined has empty profiles, so it gets no vectors."""
    texts = profiles(MinedFields())
    assert (texts.subject, texts.problem, texts.names) == ("", "", "")


def test_field_card_shows_topic_areas_two_facts_and_snippet() -> None:
    """The card lists at most two values of each key fact."""
    card = field_card(mined(), "the thread starts")
    assert card.splitlines()[0] == "429s after burst limit change"
    assert "areas: API and SDK, billing and pricing" in card
    assert "fix: idempotency keys; raise burst |" in card
    assert "revert" not in card
    assert card.endswith("text: the thread starts…")


def test_plain_card_and_source_line() -> None:
    """Unmined documents show their title; sources read as kinds, with the channel."""
    assert "(no mined fields)" in plain_card("Quarterly plan\nmore", "text")
    assert source_line("slack", "eng-releases") == "chat thread #eng-releases"
    assert source_line("gmail", None) == "email"
