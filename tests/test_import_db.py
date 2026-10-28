"""Tests for converting loaded data into database rows."""

from train_rag.import_db import to_document_record, to_question_record
from train_rag.models import Artifact, Question, QuestionType, SourceType


def test_document_record_keeps_provenance_and_original_size() -> None:
    """Sizes describe the original text, and NUL is stored as a visible escape."""
    artifact = Artifact(
        id="dsid_1",
        source_type=SourceType.SLACK,
        title="eng-runtime",
        content="set delimiter to '\x00' — done",
        source_file="data/documents/test.parquet",
        row_index=7,
    )

    record = to_document_record("enterprise_rag_bench", artifact)

    assert record.doc_id == "dsid_1"
    assert record.content == "set delimiter to '\\x00' — done"
    assert record.char_count == 27
    assert record.byte_size == 29
    assert (record.source_file, record.row_index) == ("data/documents/test.parquet", 7)


def test_question_record_stores_lists() -> None:
    """Question lists are stored as plain strings."""
    question = Question(
        id="qst_0001",
        question_type=QuestionType.BASIC,
        text="What changed?",
        gold_answer="The driver.",
        answer_facts=["The driver changed."],
        source_types=[SourceType.SLACK],
        expected_doc_ids=["dsid_1"],
    )

    record = to_question_record("enterprise_rag_bench", question)

    assert record.question_type == "basic"
    assert record.source_types == ["slack"]
    assert record.expected_doc_ids == ["dsid_1"]
